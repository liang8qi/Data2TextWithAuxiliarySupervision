#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import sys
import json

import torch

import onmt.io
import opts


def makedir(path):
    path = os.path.split(path)[0]
    if not os.path.exists(path):
        os.makedirs(path)


def check_existing_pt_files(opt):
    # We will use glob.glob() to find sharded {train|valid}.[0-9]*.pt
    # when training, so check to avoid tampering with existing pt files
    # or mixing them up.
    for t in ['train', 'valid', 'vocab']:
        pattern = opt.save_data + '.' + t + '*.pt'
        if glob.glob(pattern):
            sys.stderr.write("Please backup exisiting pt file: %s, "
                             "to avoid tampering!\n" % pattern)
            sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    makedir(opt.save_data)

    check_existing_pt_files(opt)

    return opt


def build_save_text_dataset_in_shards(src_corpus, tgt_corpus, fields,
                                      corpus_type, opt, pointers):
    '''
    Divide the big corpus into shards, and build dataset separately.
    This is currently only for data_type=='text'.

    The reason we do this is to avoid taking up too much memory due
    to sucking in a huge corpus file.

    To tackle this, we only read in part of the corpus file of size
    `max_shard_size`(actually it is multiples of 64 bytes that equals
    or is slightly larger than this size), and process it into dataset,
    then write it to disk along the way. By doing this, we only focus on
    part of the corpus at any moment, thus effectively reducing memory use.
    According to test, this method can reduce memory footprint by ~50%.

    Note! As we process along the shards, previous shards might still
    stay in memory, but since we are done with them, and no more
    reference to them, if there is memory tight situation, the OS could
    easily reclaim these memory.

    If `max_shard_size` is 0 or is larger than the corpus size, it is
    effectively preprocessed into one dataset, i.e. no sharding.

    NOTE! `max_shard_size` is measuring the input corpus size, not the
    output pt file size. So a shard pt file consists of examples of size
    2 * `max_shard_size`(source + target).
    '''

    corpus_size = os.path.getsize(src_corpus)
    if corpus_size > 10 * (1024**2) and opt.max_shard_size == 0:
        print("Warning. The corpus %s is larger than 10M bytes, you can "
              "set '-max_shard_size' to process it by small shards "
              "to use less memory." % src_corpus)

    if opt.max_shard_size != 0:
        print(' * divide corpus into shards and build dataset separately'
              '(shard_size = %d bytes).' % opt.max_shard_size)
    if opt.cal_indices:
        if type(opt.process_hier_meta) == str:
            with open(opt.process_hier_meta, "r") as f:
                opt.process_hier_meta = json.load(f)

    ret_list = []
    src_iter = onmt.io.ShardedTextCorpusIterator(
                src_corpus, opt.src_seq_length_trunc,
                "src", opt.max_shard_size, cal_indices=opt.cal_indices,
                hier_meta=opt.process_hier_meta)

    tgt_iter = onmt.io.ShardedTextCorpusIterator(
                tgt_corpus, opt.tgt_seq_length_trunc,
                "tgt", opt.max_shard_size,
                assoc_iter=src_iter)

    index = 0
    while not src_iter.hit_end():
        index += 1
        dataset = onmt.io.TextDataset(
                fields, src_iter, tgt_iter,
                src_iter.num_feats,
                src_seq_length=opt.src_seq_length,
                tgt_seq_length=opt.tgt_seq_length,
                dynamic_dict=opt.dynamic_dict, pointers_file=pointers)

        # We save fields in vocab.pt seperately, so make it empty.
        dataset.fields = []

        pt_file = "{:s}.{:s}.{:d}.pt".format(
                opt.save_data, corpus_type, index)
        print(" * saving %s datasets shard to %s." % (corpus_type, pt_file))
        torch.save(dataset, pt_file)

        ret_list.append(pt_file)

    return ret_list


def build_save_dataset(corpus_type, fields, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src
        tgt_corpus = opt.train_tgt
        pointers = opt.train_ptr
    else:
        src_corpus = opt.valid_src
        tgt_corpus = opt.valid_tgt
        pointers = None

    assert opt.data_type == 'text'
    # Currently we only do preprocess sharding for corpus: data_type=='text'.
    return build_save_text_dataset_in_shards(
            src_corpus, tgt_corpus, fields,
            corpus_type, opt, pointers=pointers)


def build_save_vocab(train_dataset, fields, opt):
    fields = onmt.io.build_vocab(train_dataset, fields, opt.data_type,
                                 opt.share_vocab,
                                 opt.src_vocab_size,
                                 opt.src_words_min_frequency,
                                 opt.tgt_vocab_size,
                                 opt.tgt_words_min_frequency)

    # Can't save fields, so remove/reconstruct at training time.
    vocab_file = opt.save_data + '.vocab.pt'
    vocab = onmt.io.save_fields_to_vocab(fields)
    torch.save(vocab, vocab_file)


def main():
    opt = parse_args()

    print("Extracting features...")
    # feature num except first element - value
    src_nfeats = onmt.io.get_num_features(opt.data_type, opt.train_src, 'src')

    tgt_nfeats = onmt.io.get_num_features(opt.data_type, opt.train_tgt, 'tgt')
    print(" * number of source features: %d." % src_nfeats)
    # print(" * number of source features- stage 2: %d." % src_nfeats2)
    print(" * number of target features: %d." % tgt_nfeats)

    print("Building `Fields` object...")

    fields = onmt.io.get_fields(opt.data_type, src_nfeats, tgt_nfeats, cal_indices=opt.cal_indices)

    print("Building & saving training datasets...")
    train_dataset_files = build_save_dataset('train', fields, opt)

    print("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt)
    # print("fields is {}".format(type(fields['src1'].vocab)))
    # assert False
    print("Building & saving validation datasets...")
    build_save_dataset('valid', fields, opt)


if __name__ == "__main__":
    main()
