# -*- coding: utf-8 -*-

from collections import Counter
from itertools import chain
import io
import codecs
import sys
import re
import numpy as np

import torch
import torchtext

from onmt.Utils import aeq
from onmt.io.BoxField import BoxField, CharField
from onmt.io.DatasetBase import (ONMTDatasetBase, UNK_WORD,
                                 PAD_WORD, BOS_WORD, EOS_WORD)

from auxiliary.utils import _sort_tables

from auxiliary.sort_algorithm import arg_sort


PAD_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3


class TextDataset(ONMTDatasetBase):
    """ Dataset for data_type=='text'

        Build `Example` objects, `Field` objects, and filter_pred function
        from text corpus.

        Args:
            fields (dict): a dictionary of `torchtext.datasets.Field`.
                Keys are like 'src', 'tgt', 'src_map', and 'alignment'.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            src_seq_length (int): maximum source sequence length.
            tgt_seq_length (int): maximum target sequence length.
            dynamic_dict (bool): create dynamic dictionaries?
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """
    def __init__(self, fields, src_examples_iter, tgt_examples_iter,
                 num_src_feats=0, num_tgt_feats=0,
                 src_seq_length=0, tgt_seq_length=0,
                 dynamic_dict=True, use_filter_pred=True, pointers_file=None):
        self.data_type = 'text'

        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.src_vocabs = []

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.
        pointers = None
        if pointers_file is not None:
            with open(pointers_file) as f:
                content = f.readlines()
            pointers = [x.strip() for x in content]

        if tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        if dynamic_dict and src_examples_iter is not None:
            examples_iter = self._dynamic_dict(examples_iter, pointers)

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]

        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        # example_values = tuple(ex_list)

        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        src_size = 0

        out_examples = []
        for ex_values in example_values:
            # object of "torchtext.datasets.Example"
            example = self._construct_example_fromlist(
                ex_values, out_fields)

            src_size += len(example.src)
            out_examples.append(example)

        print("average src size", src_size / len(out_examples),
              len(out_examples))

        def filter_pred(example):
            return 0 < len(example.src) <= src_seq_length \
                   and (pointers_file is None or 1 < example.ptrs.size(0))

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(TextDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

    @staticmethod
    def extract_text_features(tokens):
        """
        Args:
            tokens: A list of tokens, where each token consists of a word,
                optionally followed by u"￨"-delimited features.
        Returns:
            A sequence of words, a sequence of features, num of features, and a sequence of chars of words (tuple).
        """
        if not tokens:
            return [], [], -1
        split_tokens = [token.split(u"￨") for token in tokens]
        split_tokens = [token for token in split_tokens if token[0]]
        token_size = len(split_tokens[0])

        assert all(len(token) == token_size for token in split_tokens), \
            "all words must have the same number of features"
        words_and_features = list(zip(*split_tokens))
        words = words_and_features[0]

        features = words_and_features[1:]

        return words, features, token_size - 1

    @staticmethod
    def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
            blank = []
            fill = []
            index = batch.indices.data[b]
            src_vocab = src_vocabs[index]

            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).type_as(batch.indices.data)
                fill = torch.Tensor(fill).type_as(batch.indices.data)

                scores[:, b].index_add_(1, fill,
                                        scores[:, b].index_select(1, blank))
                scores[:, b].index_fill_(1, blank, 1e-10)

        return scores

    @staticmethod
    def make_text_examples_nfeats_tpl(path, truncate, side, cal_indices=False, hier_meta=None):
        """
        Args:
            path (str): location of a src or tgt file.
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".
            cal_indices:
            hier_meta:
        Returns:
            (example_dict iterator, num_feats) tuple.
        """
        assert side in ['src', 'tgt']

        if path is None:
            return (None, 0)

        # All examples have same number of features, so we peek first one
        # to get the num_feats.
        examples_nfeats_iter = \
            TextDataset.read_text_file(path, truncate, side, cal_indices=cal_indices, hier_meta=hier_meta)

        first_ex = next(examples_nfeats_iter)
        num_feats = first_ex[1]

        # Chain back the first element - we only want to peek it.
        examples_nfeats_iter = chain([first_ex], examples_nfeats_iter)
        examples_iter = (ex for ex, nfeats in examples_nfeats_iter)

        return (examples_iter, num_feats)

    @staticmethod
    def read_text_file(path, truncate, side, cal_indices=False, hier_meta=None):
        """
        Args:
            path (str): location of a src or tgt file.
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Yields:
            (word, features, nfeat) triples for each line.
        """
        with codecs.open(path, "r", "utf-8") as corpus_file:
            for i, line in enumerate(corpus_file):
                line = line.strip().split()
                if truncate:
                    line = line[:truncate]

                words, feats, n_feats= \
                    TextDataset.extract_text_features(line)

                example_dict = {side: words, "indices": i}

                if side == 'tgt1':
                    assert False
                    example_dict = {side: words, 'tgt1_planning': [int(word) for word in words], "indices": i}
                if feats:
                    prefix = side + "_feat_"
                    example_dict.update((prefix + str(j), f)
                                        for j, f in enumerate(feats))

                if cal_indices:
                    home_col_indices, home_row_indices, vis_col_indices, \
                    vis_row_indices, team_col_indices, team_row_indices = _sort_tables(words, hier_meta)

                    example_dict["home_col_indices"] = home_col_indices
                    example_dict["home_row_indices"] = home_row_indices
                    example_dict["vis_col_indices"] = vis_col_indices
                    example_dict["vis_row_indices"] = vis_row_indices
                    example_dict["team_col_indices"] = team_col_indices
                    example_dict["team_row_indices"] = team_row_indices

                yield example_dict, n_feats


    @staticmethod
    def get_fields(n_src_features, n_tgt_features, cal_indices=False):
        """
        Args:
            n_src_features (int): the number of source features to
                create `torchtext.datasets.Field` for.
            n_tgt_features (int): the number of target features to
                create `torchtext.datasets.Field` for.

        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        fields["src"] = BoxField(
            sequential=False,
            init_token=BOS_WORD,
            eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        for j in range(n_src_features):
            fields["src_feat_" + str(j)] = \
                BoxField(sequential=False, pad_token=PAD_WORD)
        if cal_indices:
            fields["home_col_indices"] = torchtext.data.Field(use_vocab=False, sequential=False,
                                                              tensor_type=torch.LongTensor)
            fields["home_row_indices"] = torchtext.data.Field(use_vocab=False, sequential=False,
                                                              tensor_type=torch.LongTensor)
            fields["vis_col_indices"] = torchtext.data.Field(use_vocab=False, sequential=False,
                                                             tensor_type=torch.LongTensor)
            fields["vis_row_indices"] = torchtext.data.Field(use_vocab=False, sequential=False,
                                                             tensor_type=torch.LongTensor)
            fields["team_col_indices"] = torchtext.data.Field(use_vocab=False, sequential=False,
                                                              tensor_type=torch.LongTensor)
            fields["team_row_indices"] = torchtext.data.Field(use_vocab=False, sequential=False,
                                                              tensor_type=torch.LongTensor)
        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        def make_src(data, vocab, is_train):

            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.zeros(src_size, len(data), src_vocab_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[j, i, t] = 1
            return alignment

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            postprocessing=make_src, sequential=False)

        def make_tgt(data, vocab, is_train):
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[:sent.size(0), i] = sent
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            postprocessing=make_tgt, sequential=False)

        def make_pointer(data, vocab, is_train):
            if is_train:
                src_size = max([t[-2][0] for t in data])
                tgt_size = max([t[-1][0] for t in data])
                #format of datasets is tgt_len, batch, src_len
                alignment = torch.zeros(tgt_size+2, len(data), src_size).long()  #+2 for bos and eos
                for i, sent in enumerate(data):
                    for j, t in enumerate(sent[:-2]):   #only iterate till the third-last row
                        # as the last two rows contains lengths of src and tgt
                        for k in range(1, t[t.size(0)-1]):   # iterate from index 1 as index 0 is tgt position
                            alignment[t[0]+1][i][t[k]] = 1  # +1 to accommodate bos
                return alignment
            else:
                return torch.zeros(50, 5, 602).long()

        fields["ptrs"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            postprocessing=make_pointer, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            sequential=False)
        return fields

    @staticmethod
    def get_num_features(corpus_file, side):
        """
        Peek one line and get number of features of it.
        (All lines must have same number of features).
        For text corpus, both sides are in text form, thus
        it works the same.

        Args:
            corpus_file (str): file path to get the features.
            side (str): 'src' or 'tgt'.

        Returns:
            number of features on `side`.
        """
        with codecs.open(corpus_file, "r", "utf-8") as cf:
            f_line = cf.readline().strip().split()
            _, _, num_feats = TextDataset.extract_text_features(f_line)

        return num_feats

    # Below are helper functions for intra-class use only.
    def _dynamic_dict(self, examples_iter, pointers=None):
        loop_index = -1
        for example in examples_iter:
            src = example["src"]
            loop_index += 1
            src_vocab = torchtext.vocab.Vocab(Counter(src),
                                              specials=[UNK_WORD, PAD_WORD])
            self.src_vocabs.append(src_vocab)
            # Mapping source tokens to indices in the dynamic dict.
            src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
            example["src_map"] = src_map

            if "tgt" in example:
                tgt = example["tgt"]
                mask = torch.LongTensor(
                    [0] + [src_vocab.stoi[w] for w in tgt] + [0])

                example["alignment"] = mask

                if pointers is not None:
                    pointer_entries = pointers[loop_index].split()
                    # tmp = pointer_entries
                    pointer_entries = [int(entry.split(",")[0]) for entry in pointer_entries]

                    mask = torch.LongTensor([0] + [src_vocab.stoi[w] if i in pointer_entries
                                                   else src_vocab.stoi[UNK_WORD] for i, w in enumerate(tgt)] + [0])

                    example["alignment"] = mask
                    max_len = 0
                    line_tuples = []
                    for pointer in pointers[loop_index].split():
                        val = [int(entry) for entry in pointer.split(",")]
                        if len(val)>max_len:
                            max_len = len(val)
                        line_tuples.append(val)
                    num_rows = len(line_tuples)+2   #+2 for storing the length of the source and target sentence
                    ptrs = torch.zeros(num_rows, max_len+1).long()  #last col is for storing the size of the row
                    for j in range(ptrs.size(0)-2): #iterating until row-1 as row contains the length of the sentence
                        for k in range(len(line_tuples[j])):
                            ptrs[j][k]=line_tuples[j][k]
                        ptrs[j][max_len] = len(line_tuples[j])
                    ptrs[ptrs.size(0)-2][0] = len(src)
                    ptrs[ptrs.size(0)-1][0] = len(tgt)
                    example["ptrs"] = ptrs
                else:
                    example["ptrs"] = None
            yield example


class ShardedTextCorpusIterator(object):
    """
    This is the iterator for text corpus, used for sharding large text
    corpus into small shards, to avoid hogging memory.

    Inside this iterator, it automatically divides the corpus file into
    shards of size `shard_size`. Then, for each shard, it processes
    into (example_dict, n_features) tuples when iterates.
    """
    def __init__(self, corpus_path, line_truncate, side, shard_size,
                 assoc_iter=None, cal_indices=False, hier_meta=None):
        """
        Args:
            corpus_path: the corpus file path.
            line_truncate: the maximum length of a line to read.
                            0 for unlimited.
            side: "src" or "tgt".
            shard_size: the shard size, 0 means not sharding the file.
            assoc_iter: if not None, it is the associate iterator that
                        this iterator should align its step with.
        """
        try:
            # The codecs module seems to have bugs with seek()/tell(),
            # so we use io.open().
            self.corpus = io.open(corpus_path, "r", encoding="utf-8")
        except IOError:
            sys.stderr.write("Failed to open corpus file: %s" % corpus_path)
            sys.exit(1)

        self.line_truncate = line_truncate
        self.side = side
        self.shard_size = shard_size
        self.assoc_iter = assoc_iter
        self.last_pos = 0
        self.line_index = -1
        self.eof = False
        self.cal_indices = cal_indices
        self.hier_meta = hier_meta

    def __iter__(self):
        """
        Iterator of (example_dict, nfeats).
        On each call, it iterates over as many (example_dict, nfeats) tuples
        until this shard's size equals to or approximates `self.shard_size`.
        """
        iteration_index = -1
        if self.assoc_iter is not None:
            # We have associate iterator, just yields tuples
            # util we run parallel with it.
            while self.line_index < self.assoc_iter.line_index:
                line = self.corpus.readline()
                if line == '':
                    raise AssertionError(
                        "Two corpuses must have same number of lines!")

                self.line_index += 1
                iteration_index += 1
                yield self._example_dict_iter(line, iteration_index)

            if self.assoc_iter.eof:
                self.eof = True
                self.corpus.close()
        else:
            # Yield tuples util this shard's size reaches the threshold.
            self.corpus.seek(self.last_pos)
            while True:
                if self.shard_size != 0 and self.line_index % 64 == 0:
                    # This part of check is time consuming on Py2 (but
                    # it is quite fast on Py3, weird!). So we don't bother
                    # to check for very line. Instead we chekc every 64
                    # lines. Thus we are not dividing exactly per
                    # `shard_size`, but it is not too much difference.
                    cur_pos = self.corpus.tell()
                    if cur_pos >= self.last_pos + self.shard_size:
                        self.last_pos = cur_pos
                        raise StopIteration
                        # return

                line = self.corpus.readline()
                if line == '':
                    self.eof = True
                    self.corpus.close()
                    break

                self.line_index += 1
                iteration_index += 1

                yield self._example_dict_iter(line, iteration_index)

    def hit_end(self):
        return self.eof

    @property
    def num_feats(self):
        # We peek the first line and seek back to
        # the beginning of the file.
        saved_pos = self.corpus.tell()

        line = self.corpus.readline().split()
        if self.line_truncate:
            line = line[:self.line_truncate]
        _, _, self.n_feats = TextDataset.extract_text_features(line)

        self.corpus.seek(saved_pos)

        return self.n_feats

    def _example_dict_iter(self, line, index):
        line = line.split()

        if self.line_truncate:
            line = line[:self.line_truncate]

        words, feats, n_feats = TextDataset.extract_text_features(line)

        example_dict = {self.side: words, "indices": index}

        if feats:
            # All examples must have same number of features.
            aeq(self.n_feats, n_feats)

            prefix = self.side + "_feat_"
            example_dict.update((prefix + str(j), f)
                                for j, f in enumerate(feats))

        if self.cal_indices:
            home_col_indices, home_row_indices, vis_col_indices, \
            vis_row_indices, team_col_indices, team_row_indices = self._sort_tables(words)
            example_dict["home_col_indices"] = home_col_indices
            example_dict["home_row_indices"] = home_row_indices
            example_dict["vis_col_indices"] = vis_col_indices
            example_dict["vis_row_indices"] = vis_row_indices
            example_dict["team_col_indices"] = team_col_indices
            example_dict["team_row_indices"] = team_row_indices

        return example_dict

    def _sort_tables(self, words):
        #
        home_start = int(self.hier_meta['home_ply_start'])
        home_end = int(self.hier_meta['home_ply_end']) + 1
        home_ply_kw_num = int(self.hier_meta['home_ply_kw_num'])
        home_ply_num = int(self.hier_meta['home_ply_num'])

        home_col_indices, home_row_indices = self._sort_one_table(words, home_start, home_end,
                                                                  home_ply_num, home_ply_kw_num)
        # print(home_indices.shape)
        # print((home_ply_kw_num, home_ply_num))
        assert len(home_col_indices) == home_ply_num*home_ply_kw_num
        assert len(home_row_indices) == home_ply_num*home_ply_kw_num
        #
        vis_ply_start = int(self.hier_meta['vis_ply_start'])
        vis_ply_end = int(self.hier_meta['vis_ply_end']) + 1
        vis_ply_kw_num = int(self.hier_meta['vis_ply_kw_num'])
        vis_ply_num = int(self.hier_meta['vis_ply_num'])

        vis_col_indices, vis_row_indices = self._sort_one_table(words, vis_ply_start, vis_ply_end,
                                                                vis_ply_num, vis_ply_kw_num)
        # print(vis_indices.shape)
        # print((vis_ply_num,  vis_ply_kw_num))
        assert len(vis_col_indices) == vis_ply_num*vis_ply_kw_num
        assert len(vis_row_indices) == vis_ply_num * vis_ply_kw_num

        #
        team_start = int(self.hier_meta['team_start'])
        team_end = int(self.hier_meta['team_end']) + 1
        team_kw_num = int(self.hier_meta['team_kw_num'])
        team_num = int(self.hier_meta['team_num'])

        team_col_indices, team_row_indices = self._sort_one_table(words, team_start, team_end, team_num, team_kw_num)
        assert len(team_col_indices) == team_num*team_kw_num
        assert len(team_row_indices) == team_num * team_kw_num

        return home_col_indices, home_row_indices, vis_col_indices, vis_row_indices, team_col_indices, team_row_indices

    @staticmethod
    def _sort_one_table(words, start, end, row_num, kw_num, col_descending=True,
                        row_descending=False):
        """

        :param words:
        :param start:
        :param end:
        :param row_num:
        :param kw_num:
        :param axis: 1 mean sort col
        :return:
        """
        table = list(words)[start:end]
        table = np.array(table).reshape(row_num, kw_num).transpose()

        for i in range(kw_num):
            for j in range(row_num):

                try:
                    num = int(table[i][j])
                except:
                    if table[i][j] == 'N/A':
                        num = -1
                    else:
                        num = 0

                table[i][j] = num
        float_table = table.astype(np.float)

        # print(float_table.shape)
        # print(float_table[:, 0].shape)
        # assert False
        # print("float_table is {}".format(float_table))

        if col_descending:
            col_indices = np.argsort(-float_table, axis=1)
        else:
            col_indices = np.argsort(float_table, axis=1)
        """
        col_indices, _ = arg_sort(float_table, descending=col_descending)
        """
        ranking = np.zeros(col_indices.shape, dtype=np.long)
        kw, n_row = col_indices.shape
        for i in range(kw):
            for j in range(n_row):
                pos = col_indices[i][j]
                ranking[i][pos] = j
        # print("ranking is {}".format(ranking.shape))

        if row_descending:  # 降序
            row_indices = np.argsort(-ranking, axis=0)
        else:
            row_indices = np.argsort(ranking, axis=0)
        """
        row_indices, _ = arg_sort(ranking, axis=0, descending=row_descending)
        """
        col_indices = col_indices.transpose().reshape(row_num * kw_num)
        row_indices = row_indices.transpose().reshape(row_num * kw_num)

        col_indices = tuple(col_indices)
        row_indices = tuple(row_indices)

        return col_indices, row_indices

    def calcu_plan(self, words):
        home_start = int(self.hier_meta['home_ply_start'])
        home_end = int(self.hier_meta['home_ply_end']) + 1
        home_ply_kw_num = int(self.hier_meta['home_ply_kw_num'])
        home_ply_num = int(self.hier_meta['home_ply_num'])

        vis_ply_start = int(self.hier_meta['vis_ply_start'])
        vis_ply_end = int(self.hier_meta['vis_ply_end']) + 1
        vis_ply_kw_num = int(self.hier_meta['vis_ply_kw_num'])
        vis_ply_num = int(self.hier_meta['vis_ply_num'])

        team_start = int(self.hier_meta['team_start'])
        team_end = int(self.hier_meta['team_end']) + 1
        team_kw_num = int(self.hier_meta['team_kw_num'])
        team_num = int(self.hier_meta['team_num'])

        home_player_indices = []
        vis_player_indices = []
        team_indices = []
        all_indices = []

        for plan_index in words:
            plan_index = int(plan_index)
            if home_start <= plan_index <= home_end:
                player_id = (plan_index - home_start - 1) // home_ply_kw_num
                if player_id not in home_player_indices:
                    home_player_indices.append(player_id)
                if player_id not in all_indices:
                    all_indices.append(player_id)
            elif vis_ply_start <= plan_index <= vis_ply_end:
                player_id = (plan_index - vis_ply_start - 1) // vis_ply_kw_num
                if player_id not in vis_player_indices:
                    vis_player_indices.append(player_id)
                all_player_id = player_id + home_ply_num
                if all_player_id not in all_indices:
                    all_indices.append(all_player_id)
            elif team_start <= plan_index <= team_end:
                player_id = (plan_index - team_start - 1) // team_kw_num
                if player_id not in team_indices:
                    team_indices.append(player_id)

                all_player_id = player_id + home_ply_num + vis_ply_num
                if all_player_id not in all_indices:
                    all_indices.append(all_player_id)
            else:
                assert False

        assert len(home_player_indices) <= home_ply_num
        assert len(vis_player_indices) <= vis_ply_num
        assert len(team_indices) <= team_num

        for i in range(home_ply_num):
            if i not in home_player_indices:
                home_player_indices.append(i)

        for i in range(vis_ply_num):
            if i not in vis_player_indices:
                vis_player_indices.append(i)

        for i in range(team_num):
            if i not in team_indices:
                team_indices.append(i)

        for i in range(home_ply_num + vis_ply_num + team_num):
            if i not in all_indices:
                all_indices.append(i)

        return home_player_indices, vis_player_indices, team_indices, all_indices

