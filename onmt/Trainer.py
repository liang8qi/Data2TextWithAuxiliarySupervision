# -*- coding: utf-8 -*-
from __future__ import division

"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.modules


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = float(loss)

        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

        self.num_ranking_loss = 0.0
        self.num_ranking_cnt = 0
        self.imp_ranking_loss = 0.0
        self.imp_ranking_cnt = 0

        self.num_ranking_n_correct = 0
        self.num_ranking_n_ex = 0

        self.imp_ranking_n_correct = 0
        self.imp_ranking_n_ex = 0

        self.updated_cnt = 0

    def update(self, stat):
        self.loss += float(stat.loss)
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.updated_cnt += 1

    def update_num_ranking_loss(self, num_ranking_loss, cnt):
        self.num_ranking_loss += num_ranking_loss
        self.num_ranking_cnt += cnt

    def update_imp_ranking_loss(self, imp_ranking_loss, cnt):
        self.imp_ranking_loss += imp_ranking_loss
        self.imp_ranking_cnt += cnt

    def update_num_ranking_metrics(self, n_correct, n_ex):
        self.num_ranking_n_correct += n_correct
        self.num_ranking_n_ex += n_ex

    def update_imp_ranking_metrics(self, n_correct, n_ex):
        self.imp_ranking_n_correct += n_correct
        self.imp_ranking_n_ex += n_ex

    def cal_num_ranking_accuracy(self):
        return float(self.num_ranking_n_correct) / float(self.num_ranking_n_ex)

    def cal_imp_ranking_accuracy(self):
        return float(self.imp_ranking_n_correct) / float(self.imp_ranking_n_ex)

    def accuracy(self):
        n_words = self.n_words if self.n_words > 0 else 1
        return 100 * (float(self.n_correct) / float(n_words))

    def ppl(self):
        denominator = self.n_words if self.n_words > 0 else 1
        return math.exp(min(float(self.loss) / float(denominator), 100))

    def xent(self):
        """ compute cross entropy """
        denominator = self.n_words if self.n_words > 0 else 1
        return float(self.loss) / float(denominator)

    def cal_num_ranking_loss(self):
        if self.num_ranking_cnt == 0:
            return 0.0
        return float(self.num_ranking_loss) / float(self.num_ranking_cnt)

    def cal_imp_ranking_loss(self):
        return float(self.imp_ranking_loss) / float(self.imp_ranking_cnt)

    def cal_ranking_loss(self):
        col_loss = 0.0
        row_loss = 0.0
        if self.num_ranking_cnt > 0:
            col_loss = self.cal_num_ranking_loss()
        if self.imp_ranking_cnt > 0:
            row_loss = self.cal_imp_ranking_loss()

        n = 0
        total_loss = [col_loss, row_loss]
        for loss in total_loss:
            if loss > 0:
                n += 1

        return sum(total_loss) / float(n)

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        print_para = "Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.4f; xnet: %4.4f; "
        t = self.elapsed_time()

        print_val = (epoch, batch, n_batches, self.accuracy(), self.ppl(), self.xent())

        if self.num_ranking_loss > 0 or self.imp_ranking_loss > 0:
            print_para += " ranking_loss: %4.4f;"
            print_val += (self.cal_ranking_loss(),)
            if self.num_ranking_cnt > 0:
                print_para = print_para + " num_ranking_loss: %4.4f;"
                print_val += (self.cal_num_ranking_loss(),)
                if self.num_ranking_n_correct > 0:
                    print_para = print_para + " num_ranking_acc: %.5f;"
                    print_val += (self.cal_num_ranking_accuracy(),)

            if self.imp_ranking_cnt > 0:
                print_para = print_para + "imp_ranking_loss: %4.4f;"
                print_val += (self.cal_imp_ranking_loss(),)
                if self.imp_ranking_n_correct > 0:
                    print_para = print_para + " imp_ranking_acc: %.5f;"
                    print_val += (self.cal_imp_ranking_accuracy(),)

        print_para = print_para + " %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed"
        print_val += (self.n_src_words / (t + 1e-5), self.n_words / (t + 1e-5), time.time() - start)

        print(print_para % print_val)
        sys.stdout.flush()

    def log(self, prefix, experiment, lr_2, lr_1=None):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper", self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr2", lr_2)
        if lr_1 is not None:
            experiment.add_scalar_value(prefix + "_lr1", lr_1)
        experiment.add_scalar_value(prefix + '_xnet', self.xent())
        if self.ranking_loss > 0:
            experiment.add_scalar_value(prefix + '_ranking_loss', self.cal_ranking_loss())

    def log_tensorboard(self, prefix, writer, lr_2, epoch, lr_1=None):
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), epoch)
        writer.add_scalar(prefix + "/ppl", self.ppl(), epoch)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), epoch)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, epoch)
        writer.add_scalar(prefix + "/lr_2", lr_2, epoch)
        if lr_1 is not None:
            writer.add_scalar(prefix + "/lr_1", lr_1, epoch)
        if self.num_ranking_loss > 0:
            writer.add_scalar(prefix + '/num_ranking_loss', self.cal_num_ranking_loss(), epoch)
        if self.imp_ranking_loss > 0:
            writer.add_scalar(prefix + '/imp_ranking_loss', self.cal_imp_ranking_loss(), epoch)


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
    """

    def __init__(self, model, model2, train_loss, valid_loss, train_loss2, valid_loss2, optim, optim2,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1, cuda=False,
                 enable_number_ranking=False, enable_importance_ranking=False, ptr_train_loss=None, alpha_nr=1.0, only_ptr=False,
                 normalize_by_length=False, alpha_ir=1.0):
        # Basic attributes.
        self.model = model
        self.model2 = model2
        self.train_loss = train_loss
        self.ptr_train_loss = ptr_train_loss
        self.valid_loss = valid_loss
        self.train_loss2 = train_loss2
        self.valid_loss2 = valid_loss2
        self.optim = optim
        self.optim2 = optim2
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.cuda = cuda
        self.enable_number_ranking = enable_number_ranking
        self.enable_importance_ranking = enable_importance_ranking
        self.alpha_nr = alpha_nr
        self.alpha_ir = alpha_ir
        self.only_ptr = only_ptr
        self.normalize_by_length = normalize_by_length

        assert (grad_accum_count > 0)
        if grad_accum_count > 1:
            assert (self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        if self.model is not None:
            self.model.train()
        self.model2.train()

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training datasets iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = report_stats = None
        total_stats2 = Statistics()
        report_stats2 = Statistics()
        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        for i, batch in enumerate(train_iter):
            # if i % 10 == 0:
            #     torch.cuda.empty_cache()
            cur_dataset = train_iter.get_cur_dataset()
            if self.train_loss is not None:
                self.train_loss.cur_dataset = cur_dataset
            self.train_loss2.cur_dataset = cur_dataset

            loss_pad = self.train_loss if self.train_loss is not None else self.train_loss2
            true_batchs.append(batch)
            accum += 1
            # what is batch.tgt?
            if self.norm_method == "tokens":
                assert False
                num_tokens = batch.tgt[1:].data.view(-1) \
                    .ne(loss_pad.padding_idx).sum()
                normalization += num_tokens
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:
                self._gradient_accumulation_basic_encdec(true_batchs, total_stats2, report_stats2, normalization)

                if report_func is not None:
                    report_stats2 = report_func(
                        epoch, idx, num_batches,
                        total_stats2.start_time, self.optim2.lr,
                        report_stats2)

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

        if len(true_batchs) > 0:
            self._gradient_accumulation_basic_encdec(true_batchs, total_stats2, report_stats2, normalization)

            true_batchs = []

        return total_stats, total_stats2

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate datasets iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        if self.model is not None:
            self.model.eval()
            stats = Statistics()
        else:
            stats = None
        torch.cuda.empty_cache()
        self.model2.eval()
        stats2 = Statistics()
        with torch.no_grad():
            for batch in valid_iter:
                cur_dataset = valid_iter.get_cur_dataset()
                if self.model is not None:
                    self.valid_loss.cur_dataset = cur_dataset
                self.valid_loss2.cur_dataset = cur_dataset

                src = onmt.io.make_features(batch, 'src', self.data_type)
                self.tt = torch.cuda if self.cuda else torch
                src_lengths = self.tt.LongTensor(batch.src.size()[1]).fill_(batch.src.size()[0])
                src_hist = None

                tgt = onmt.io.make_features(batch, 'tgt')
                # F-prop through the model.

                if self.only_ptr:
                    col_reps, row_reps, entity_reps = self.model2.encoding(src, src_lengths)
                else:
                    outputs, attns, _, _, col_reps, row_reps, entity_reps = self.model2(src, tgt, src_lengths)

                # cal
                if self.enable_number_ranking:

                    num_ranking_loss, col_indices_list, col_argmaxs = self.training_num_ranking_decoder(col_reps=col_reps, batch=batch)
                    correct, ex_cnt = self.cal_pointer_metrics(col_argmaxs, col_indices_list)
                    stats2.update_num_ranking_metrics(correct, ex_cnt)

                    stats2.update_num_ranking_loss(num_ranking_loss.item(), 3010)
                if self.enable_importance_ranking:
                    imp_ranking_loss, row_indices_list, row_argmaxs = self.training_imp_ranking_decoder(row_reps=row_reps,
                                                                                                batch=batch)

                    row_correct, row_ex_cnt = self.cal_pointer_metrics(row_argmaxs, row_indices_list)
                    stats2.update_imp_ranking_metrics(row_correct, row_ex_cnt)

                    stats2.update_imp_ranking_loss(imp_ranking_loss.item(), 3010)

                if not self.only_ptr:
                    # Compute loss.
                    batch_stats = self.valid_loss2.monolithic_compute_loss(
                        batch, outputs, attns, stage1=False)
                    # Update statistics.
                    stats2.update(batch_stats)


        # Set model back to training mode.
        if self.model is not None:
            self.model.train()
        self.model2.train()

        return stats, stats2

    def epoch_step(self, ppl, ppl2, epoch):
        if self.optim is not None:
            self.optim.update_learning_rate(ppl, epoch, optim_num=1)
        self.optim2.update_learning_rate(ppl2, epoch, optim_num=2)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats, valid_stats2):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        if self.model is not None:
            real_model = (self.model.module
                          if isinstance(self.model, nn.DataParallel)
                          else self.model)
            real_generator = (real_model.generator.module
                              if isinstance(real_model.generator, nn.DataParallel)
                              else real_model.generator)

            model_state_dict = real_model.state_dict()
            model_state_dict = {k: v for k, v in model_state_dict.items()
                                if 'generator' not in k}
            generator_state_dict = real_generator.state_dict()
            # onmt.io.save_fields_to_vocab(fields)
            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'vocab': onmt.io.save_fields_to_vocab(fields),
                'opt': opt,
                'epoch': epoch,
                'optim': self.optim,
            }
            torch.save(checkpoint,
                       '%s_stage1_acc_%.4f_ppl_%.4f_e%d.pt'
                       % (opt.save_model, valid_stats.accuracy(),
                          valid_stats.ppl(), epoch))

        real_model = (self.model2.module
                      if isinstance(self.model2, nn.DataParallel)
                      else self.model2)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim_2': self.optim2,
        }
        if self.optim is not None:
            checkpoint['optim1'] = self.optim
        saved_path = '%s_stage2_acc_%.4f_ppl_%.4f_%.4f_e%d.pt' % \
                     (opt.save_model, valid_stats2.accuracy() if valid_stats2 is not None else 0,
                      valid_stats2.ppl() if valid_stats2 is not None else 0,
                      valid_stats2.xent()
                      if valid_stats2 is not None else 0, epoch)
        torch.save(checkpoint, saved_path)
        print("Model has been saved at %s" % saved_path)

    def _gradient_accumulation_basic_encdec(self, true_batchs, total_stats2, report_stats2, normalization):
        if self.grad_accum_count > 1:
            assert False
            self.model.zero_grad()

        for batch in true_batchs:
            # Stage 1
            src = onmt.io.make_features(batch, 'src', self.data_type)

            self.tt = torch.cuda if self.cuda else torch
            src_lengths = self.tt.LongTensor(batch.src.size()[1]).fill_(batch.src.size()[0])

            src_hist = None
            # Stage 2
            target_size = batch.tgt.size(0)
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                # assert False
                trunc_size = target_size
            dec_state = None
            report_stats2.n_src_words += src_lengths.sum()

            if self.data_type == 'text':
                tgt_outer = onmt.io.make_features(batch, 'tgt')

            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]
                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model2.zero_grad()

                if not self.only_ptr:
                    outputs, attns, dec_state, _, col_reps, row_reps, entity_reps = \
                        self.model2(src, tgt, src_lengths, dec_state)

                if self.enable_importance_ranking or self.enable_number_ranking:
                    col_reps, row_reps, entity_reps = self.model2.encoding(src, src_lengths)
                    ranking_loss = 0.0
                if self.enable_number_ranking:

                    num_ranking_loss, _, _ = self.training_num_ranking_decoder(col_reps=col_reps, batch=batch)
                    num_ranking_loss = self.alpha_nr * num_ranking_loss

                    ranking_loss = num_ranking_loss
                    report_stats2.update_num_ranking_loss(num_ranking_loss.item(), 1)

                if self.enable_importance_ranking:
                    imp_ranking_loss, _, _ = self.training_imp_ranking_decoder(row_reps=row_reps, batch=batch)
                    imp_ranking_loss = self.alpha_ir * imp_ranking_loss
                    ranking_loss = ranking_loss + imp_ranking_loss

                    report_stats2.update_imp_ranking_loss(imp_ranking_loss.item(), 1)

                if self.enable_number_ranking or self.enable_importance_ranking:
                    ranking_loss.backward(retain_graph=True)
                
                # retain_graph is false for the final truncation
                retain_graph = (j + trunc_size) < (target_size - 1)

                # 3. Compute loss in shards for memory efficiency.
                if not self.only_ptr:
                    batch_stats = self.train_loss2.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size, normalization,
                        retain_graph=retain_graph)  # ***这里修改了retain_graph=retain_graph

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    if self.optim is not None:
                        self.optim.step()
                    self.optim2.step()
                if not self.only_ptr:
                    total_stats2.update(batch_stats)
                    report_stats2.update(batch_stats)

                # assert False
                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

    @staticmethod
    def change_shape_for_col(col_rep, indices):
        """

        :param col_rep: (kw_num, bsz, row_num, hsz)
        :param indices: (bsz, kw_num*row_num)
        :return:
        """
        kw_num, bsz, ply_num, hsz = col_rep.size()
        assert kw_num * ply_num == indices.size(-1), "{}: {}".format(kw_num * ply_num, indices.size(-1))
        indices = indices.view(bsz, ply_num, kw_num)
        indices = indices.transpose(1, 2).contiguous().view(-1, ply_num)

        return indices

    @staticmethod
    def cal_pointer_metrics(argmaxs, indices):
        cnt = 0
        total = 0
        for i in range(len(argmaxs)):
            argmaxs[i] = argmaxs[i].squeeze(2)
            correct_cnt = argmaxs[i] == indices[i]

            total += correct_cnt.size(0) * correct_cnt.size(1)
            cnt += correct_cnt.sum()

        return cnt, total

    def training_num_ranking_decoder(self, col_reps, batch):
        col_sorted_list, col_indices_list, col_argmaxs = self.model2.col_sorted(col_reps=col_reps, batch=batch)

        num_ranking_loss, _ = self.ptr_train_loss.compute(sorted_list=col_sorted_list,
                                                      indices_list=col_indices_list)

        return num_ranking_loss, col_indices_list, col_argmaxs

    def training_imp_ranking_decoder(self, row_reps, batch):
        row_sorted_list, row_indices_list, row_argmaxs = self.model2.row_sorted(row_reps=row_reps, batch=batch)

        imp_ranking_loss, _ = self.ptr_train_loss.compute(sorted_list=row_sorted_list,
                                                      indices_list=row_indices_list)
        return imp_ranking_loss, row_indices_list, row_argmaxs




