#!/usr/bin/env python

from __future__ import division, unicode_literals
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse

import math
import codecs
import torch
import time
import numpy as np

from itertools import count

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import opts
import json

from auxiliary.utils import cal, export_excel

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()


def makedir(path):
    path = os.path.split(path)[0]
    if not os.path.exists(path):
        os.makedirs(path)


makedir(opt.output)


def _report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total / words_total)))


def _report_bleu():
    import subprocess
    print()
    res = subprocess.check_output(
        "perl tools/multi-bleu.perl %s < %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(">> " + res.strip())


def _report_rouge():
    import subprocess
    res = subprocess.check_output(
        "python tools/test_rouge.py -r %s -c %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(res.strip())


def translate_one(opt, model_src, output_src, cal_indices=True):
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # Load the model.
    model = None
    model2 = None

    fields2, model2, model_opt2 = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__, model_src=model_src, stage1=False)

    fields = fields2
    if cal_indices:
        fields = onmt.io.load_fields_from_vocab(
            torch.load(model_opt2.data + '.vocab.pt'), 'text', cal_indices=cal_indices)

    model_opt = model_opt2
    print("---------The parameters are:----------")
    opt_dict = model_opt.__dict__
    for name in opt_dict.keys():
        print("{} = {}".format(name, opt_dict[name]))
    print("----------Model Structure is:-------------")
    print(model2)
    out_file = codecs.open(output_src, 'w', 'utf-8')

    data = onmt.io.build_dataset(fields, opt.data_type,
                                 opt.src, opt.src_hist, 
                                 opt.src, opt.tgt,
                                 src_dir=opt.src_dir,
                                 sample_rate=opt.sample_rate,
                                 window_size=opt.window_size,
                                 window_stride=opt.window_stride,
                                 window=opt.window,
                                 use_filter_pred=False,
                                 cal_indices=cal_indices,
                                 hier_meta=model_opt.hier_meta)
    # , cal_indices=False, hier_meta=model_opt.hier_meta

    def sort_minibatch_key(ex):
        """ Sort using length of source sentences and length of target sentence """
        #Needed for packed sequence
        return len(ex.src)

    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
    # need to be changed for rnn encoder of basic encoder-decoder framework
    data_iter = onmt.io.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        sort_key=sort_minibatch_key,
        sort_within_batch=True, shuffle=False)
    # Translator
    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)
    tgt_plan_map = None

    translator = onmt.translate.Translator(
        model, model2, fields,
        beam_size=opt.beam_size,
        n_best=opt.n_best,
        global_scorer=scorer,
        max_length=opt.max_length,
        copy_attn=model_opt.copy_attn and tgt_plan_map is None,
        cuda=opt.cuda,
        beam_trace=opt.dump_beam != "",
        min_length=opt.min_length,
        stepwise_penalty=opt.stepwise_penalty,
        is_num_ranking=False,
        is_imp_ranking=False)

    builder = onmt.translate.TranslationBuilder(
        data, translator.fields,
        opt.n_best, opt.replace_unk, has_tgt=False)

    # Statistics
    counter = count(1)
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0
    total_num_ranking_correct, total_num_ranking_ex_cnt = 0.0, 0.0
    total_imp_ranking_correct, total_imp_ranking_ex_cnt = 0.0, 0.0
    total_dld_col = 0.0
    total_dld_row = 0.0
    n_example = 0
    stage1 = opt.stage1
    for batch in data_iter:
        batch_data, col_sort_result, row_sort_result = translator.translate_batch(batch, data, stage1)
        # [total_n_corr_col, total_n_ex_col, total_dld_col], [total_n_corr_row, total_n_ex_row, total_dld_row]

        if model_opt2.enable_number_ranking or model_opt2.enable_importance_ranking:

            if col_sort_result[0] > 0:
                total_num_ranking_correct += col_sort_result[0]
                total_num_ranking_ex_cnt += col_sort_result[1]
                total_dld_col += col_sort_result[2]
            if row_sort_result[0] > 0:
                total_imp_ranking_correct += row_sort_result[0]
                total_imp_ranking_ex_cnt += row_sort_result[1]
                total_dld_row += row_sort_result[2]

        translations = builder.from_batch(batch_data, stage1)

        for trans in translations:
            pred_score_total += trans.pred_scores[0]
            pred_words_total += len(trans.pred_sents[0])
            if opt.tgt:
                gold_score_total += trans.gold_score
                gold_words_total += len(trans.gold_sent)

            if stage1:
                n_best_preds = [" ".join([str(entry) for entry in pred])
                                for pred in trans.pred_sents[:opt.n_best]]
            else:
                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:opt.n_best]]
            out_file.write('\n'.join(n_best_preds))
            out_file.write('\n')
            out_file.flush()

            if opt.verbose:
                sent_number = next(counter)
                output = trans.log(sent_number)
                os.write(1, output.encode('utf-8'))
        n_example += 1

    _report_score('PRED', pred_score_total, pred_words_total)
    if total_num_ranking_correct > 0:
        print("Col Level sort precision is {}: {}, {}".format(total_num_ranking_correct, total_num_ranking_ex_cnt,
                                                              float(total_num_ranking_correct)/float(total_num_ranking_ex_cnt)))
        print("Col level DLD is {}".format(total_dld_col/float(n_example)))

    if total_imp_ranking_correct > 0:
        print("Row Level sort precision is {}: {}, {}".format(total_imp_ranking_correct, total_imp_ranking_ex_cnt,
                                                              float(total_imp_ranking_correct)/float(total_imp_ranking_ex_cnt)))
        print("Row level DLD is {}".format(total_dld_row / float(n_example)))
    if opt.tgt:
        _report_score('GOLD', gold_score_total, gold_words_total)
        if opt.report_bleu:
            _report_bleu()
        if opt.report_rouge:
            _report_rouge()

    if opt.dump_beam:
        import json
        json.dump(translator.beam_accum,
                  codecs.open(opt.dump_beam, 'w', 'utf-8'))

def main(gold_path):
    models_src = opt.model2
    pred_file_src = opt.output
    # translate
    start_time = time.time()
    translate_one(opt, model_src=models_src, output_src=pred_file_src)
    bleu = cal(gold_path, opt.output)
    print("The bleu score is {}".format(bleu))
    print("Finished, Spending %.4f, result has been saved at %s" % (time.time() - start_time, opt.output))


if __name__ == "__main__":
    gold_path = 'ref/test.txt'
    start_time = time.time()
    main(gold_path)
    print("Spending %.4f" % (time.time()-start_time))
