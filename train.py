#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import glob
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import random

import torch
import torch.nn as nn
from torch import cuda

import onmt
import onmt.io
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import use_gpu
import opts
import numpy as np
import json

from onmt.Trainer import Trainer
# from added.ptr_decoder import PtrDecoder
from auxiliary.loss import PtrCrossEntropyLoss


parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)

opt = parser.parse_args()

if opt.hier_meta is not None:
    with open(opt.hier_meta, "r") as f:
        opt.hier_meta = json.load(f)

if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn2 = (opt.encoder_type2 == "brnn")
if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

# more reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = True

if opt.seed > 0:
    np.random.seed(opt.seed)

if opt.rnn_type == "SRU" and not opt.gpuid:
    raise AssertionError("Using SRU requires -gpuid set.")

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)

if len(opt.gpuid) > 1:
    sys.stderr.write("Sorry, multigpu isn't supported yet, coming soon!\n")
    sys.exit(1)

# Set up the Crayon logging server.
if opt.exp_host != "":
    from pycrayon import CrayonClient

    cc = CrayonClient(hostname=opt.exp_host)

    experiments = cc.get_experiment_names()
    print(experiments)
    if opt.exp in experiments:
        cc.remove_experiment(opt.exp)
    experiment = cc.create_experiment(opt.exp)

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(opt.tensorboard_log_dir, comment="Onmt")


def report_func(epoch, batch, num_batches,
                start_time, lr_2, report_stats, lr_1=None):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr_2)
        if opt.tensorboard:
            # Log the progress using the number of batches on the x-axis.
            report_stats.log_tensorboard(
                "progress", writer, lr_2, epoch * num_batches + batch, lr_1=lr_1)
        report_stats = onmt.Statistics()

    return report_stats


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def get_cur_dataset(self):
        return self.cur_dataset

    def sort_minibatch_key(self, ex):
        """ Sort using length of source sentences and length of target sentence """
        #Needed for packed sequence
        return len(ex.src)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return onmt.io.OrderedIterator(
            dataset=self.cur_dataset, batch_size=self.batch_size,
            batch_size_fn=self.batch_size_fn,
            device=self.device, train=self.is_train,
            sort_key=self.sort_minibatch_key,
            sort=False, sort_within_batch=True,
            repeat=False)


def make_dataset_iter(datasets, fields, opt, is_train=True):
    """
    This returns user-defined train/validate datasets iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """

    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_size_fn = None
    if is_train and opt.batch_type == "tokens":
        global max_src_in_batch, max_tgt_in_batch

        def batch_size_fn(new, count, sofar):
            global max_src_in_batch, max_tgt_in_batch
            if count == 1:
                max_src_in_batch = 0
                max_tgt_in_batch = 0
            max_src_in_batch = max(max_src_in_batch,  len(new.src) + 2)
            max_tgt_in_batch = max(max_tgt_in_batch,  len(new.tgt) + 1)
            src_elements = count * max_src_in_batch
            tgt_elements = count * max_tgt_in_batch
            return max(src_elements, tgt_elements)

    device = opt.gpuid[0] if opt.gpuid else -1

    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train)


def make_loss_compute(model, tgt_vocab, opt, normalize_by_length=False, stage1=True):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    if not stage1:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, opt.copy_attn_force,
            normalize_by_length)
    else:
        """
        compute = onmt.Loss.NMTLossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing, decoder_type=opt.decoder_type1)
        """
        compute = PtrCrossEntropyLoss()

    if use_gpu(opt):
        compute.cuda()

    return compute


def train_model(model, model2, fields, optim, optim2, data_type, model_opt):
    if model is not None:
        assert False
    else:
        train_loss = valid_loss = None
    if model_opt.enable_number_ranking or model_opt.enable_importance_ranking:
        ptr_train_loss = PtrCrossEntropyLoss()
    else:
        ptr_train_loss = None

    train_loss2 = make_loss_compute(model2, fields["tgt"].vocab, opt,
                                    normalize_by_length=opt.copy_loss_by_seqlength,
                                    stage1=False)
    valid_loss2 = make_loss_compute(model2, fields["tgt"].vocab, opt, stage1=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count

    cuda = False
    if opt.gpuid:
        cuda = True

    trainer = Trainer(model, model2, train_loss, valid_loss, train_loss2, valid_loss2, optim, optim2,
                      trunc_size, shard_size, data_type,
                      norm_method, grad_accum_count, cuda, enable_number_ranking=opt.enable_number_ranking,
                      enable_importance_ranking=opt.enable_importance_ranking,
                      ptr_train_loss=ptr_train_loss, alpha_nr=opt.alpha_nr,
                      only_ptr=opt.only_ptr, alpha_ir=opt.alpha_ir)

    print('\nStart training...')
    print(' * number of epochs: %d, starting from Epoch %d' %
          (opt.epochs + 1 - opt.start_epoch, opt.start_epoch))
    print(' * batch size: %d' % opt.batch_size)
    valid_stats = None
    valid_stats2 = None

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        # for epoch in range(0, opt.epochs + 1):
        print('')

        # 1. Train for one epoch on the training set.
        train_iter = make_dataset_iter(lazily_load_dataset("train"),
                                       fields, opt)

        train_stats, train_stats2 = trainer.train(train_iter, epoch, report_func)
        if train_stats is not None:
            print('Train perplexity: %g' % train_stats.ppl())
            print('Train accuracy: %g' % train_stats.accuracy())
        print('Train perplexity2: %g' % train_stats2.ppl())
        print('Train accuracy2: %g' % train_stats2.accuracy())

        if opt.training_with_val:
            # 2. Validate on the validation set.
            valid_iter = make_dataset_iter(lazily_load_dataset("valid"),
                                           fields, opt,
                                           is_train=False)
            valid_stats, valid_stats2 = trainer.validate(valid_iter)
            if valid_stats is not None:
                print('Validation perplexity: %g' % valid_stats.ppl())
                print('Validation accuracy: %g' % valid_stats.accuracy())
            print('Validation perplexity2: %g' % valid_stats2.ppl())
            print('Validation accuracy2: %g' % valid_stats2.accuracy())
            if valid_stats2.num_ranking_n_ex > 0:
                print('Validation num_ranking accuracy: %f' % valid_stats2.cal_num_ranking_accuracy())
                print("{}: {}".format(valid_stats2.num_ranking_n_ex, valid_stats2.num_ranking_n_correct))
            if valid_stats2.imp_ranking_n_ex > 0:
                print('Validation imp_ranking accuracy: %f' % valid_stats2.cal_imp_ranking_accuracy())
                print("{}: {}".format(valid_stats2.imp_ranking_n_ex, valid_stats2.imp_ranking_n_correct))

        # 3. Log to remote server.
        if opt.exp_host and train_stats is not None:
            train_stats.log("train", experiment, optim.lr)
            valid_stats.log("valid", experiment, optim.lr)
        if opt.tensorboard and train_stats is not None:
            train_stats.log_tensorboard("train", writer, optim2.lr, epoch, lr_1=None if optim is None else optim.lr)
            train_stats.log_tensorboard("valid", writer, optim2.lr, epoch, lr_1=None if optim is None else optim.lr)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats.ppl() if valid_stats is not None else None,
                           valid_stats2.ppl() if valid_stats2 is not None else None, epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats, valid_stats2)


def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    num_ranking = 0
    imp_ranking = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
        elif "num_ranking_decoder" in name:
            num_ranking += param.nelement()
        elif "imp_ranking_decoder" in name:
            imp_ranking += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)
    if num_ranking > 0:
        print("num_ranking_decoder", num_ranking)

    if imp_ranking > 0:
        print("imp_ranking_decoder:", imp_ranking)


def lazily_load_dataset(corpus_type):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        print('Loading %s dataset from %s, number of examples: %d' %
              (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one onmt.io.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        yield lazy_dataset_loader(pt, corpus_type)


def load_fields(dataset, data_type, checkpoint, cal_indices=False):
    """"
    if checkpoint is not None:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = onmt.io.load_fields_from_vocab(
            checkpoint['vocab'], data_type)
    else:
        fields = onmt.io.load_fields_from_vocab(
            torch.load(opt.data + '.vocab.pt'), data_type, cal_indices=cal_indices)
    """
    # ***
    fields = onmt.io.load_fields_from_vocab(
        torch.load(opt.data + '.vocab.pt'), data_type, cal_indices=cal_indices)

    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    if data_type == 'text' or data_type == 'box':
        print(' * vocabulary size. source = %d;target = %d' %
              (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    else:
        assert False
        print(' * vocabulary size. target = %d' %
              (len(fields['tgt'].vocab)))

    return fields


def collect_report_features(fields):
    src_features = onmt.io.collect_features(fields, side='src')
    tgt_features = onmt.io.collect_features(fields, side='tgt')

    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        print(' * tgt feature %d size = %d' % (j, len(fields[feat].vocab)))


def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')

    model1 = None
    model2 = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                   use_gpu(opt), checkpoint, stage1=False, basic_enc_dec=True)

    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        if model1 is not None:
            model1 = nn.DataParallel(model1, device_ids=opt.gpuid, dim=1)
        model2 = nn.DataParallel(model2, device_ids=opt.gpuid, dim=1)
    if model1 is not None:
        print(model1)
    print(model2)

    return model1, model2


def build_optim(model, checkpoint):
    # is_sharing meaning whether the embedding layer and ptr sharing same optim

    if opt.train_from and checkpoint is not None:
        print('Loading optimizer from checkpoint.')
        optim_name = 'optim_2'

        optim = checkpoint[optim_name]
        optim.optimizer.load_state_dict(
            checkpoint[optim_name].optimizer.state_dict())
        # optim.start_decay_at = 60
        # optim.lr = 0.08
        print("Optim Type is {}, learning_rate is {}, start_decay_at: {}".format(optim.method, optim.lr,
                                                                                  optim.start_decay_at))
    else:
        print('Making optimizer for training.')
        optim_type = opt.optim_2
        learning_rate = opt.learning_rate_2
        max_grad_norm = opt.max_grad_norm_2
        learning_rate_decay = opt.learning_rate_decay_2
        start_decay_at = opt.start_decay_at_2

        optim = onmt.Optim(
            optim_type,
            learning_rate,
            max_grad_norm,
            lr_decay=learning_rate_decay,
            start_decay_at=start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size,
            decay_every=opt.decay_every)

        print("Optim Type is {}, learning_rate is {}, start_decay_at: {}".format(optim_type, optim.lr,
                                                                                  optim.start_decay_at))

    model_name_parameters = list(model.named_parameters())

    updated_parameters = [(key, params) for key, params in model_name_parameters if params.requires_grad]
    updated_para_names = [key for key, params in updated_parameters]

    optim.set_parameters(updated_parameters)

    return optim


def main():
    print('Experiment 22-4.4 using attn_dim of 64')
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear.
        opt.start_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        model_opt = opt

    # *****
    model_opt = opt
    print("---------The parameters are:----------")
    opt_dict = opt.__dict__
    for name in opt_dict.keys():
        print("{} = {}".format(name, opt_dict[name]))

    # Peek the fisrt dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train"))
    data_type = first_dataset.data_type
    # Load fields generated from preprocess phase.
    fields = load_fields(first_dataset, data_type, checkpoint, cal_indices=opt.cal_indices)

    # Report src/tgt features.
    collect_report_features(fields)
    # Build model.
    model1, model2 = build_model(model_opt, opt, fields, checkpoint)
    if model1 is not None:
        tally_parameters(model1)
    tally_parameters(model2)
    check_save_model_path()

    # Build optimizer.
    optim1 = None
    optim2 = build_optim(model2, None)

    # Do training.
    train_model(model1, model2, fields, optim1, optim2, data_type, model_opt)

    # If using tensorboard for logging, close the writer after training.
    if opt.tensorboard:
        writer.close()


if __name__ == "__main__":
    main()
