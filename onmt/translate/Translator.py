import torch
from torch.autograd import Variable

from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance

import onmt.translate.Beam
import onmt.io


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): datasets fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, model2, fields,
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 cuda=False,
                 beam_trace=False,
                 min_length=0,
                 stepwise_penalty=False,
                 is_num_ranking=False,
                 is_imp_ranking=False):
        self.model = model
        self.model2 = model2
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.cuda = cuda
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.is_num_ranking = is_num_ranking
        self.is_imp_ranking = is_imp_ranking
        # if self.is_imp_ranking:
        #     self.is_num_ranking = True
        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate_batch(self, batch, data, stage1):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           tgt_plan_map (): mapping between tgt indices and tgt_planning


        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size

        data_type = data.data_type
        tgt = "tgt"

        vocab = self.fields[tgt].vocab
        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a):
            if isinstance(a, tuple):
                result_tuple = []
                for each_one in a:
                    if isinstance(each_one, tuple):
                        tmp_tuple = []
                        for each_tensor in each_one:
                            if each_tensor is not None:
                                tmp_tuple.append(var(each_tensor.data.repeat(1, beam_size, 1)))
                            else:
                                tmp_tuple.append(None)
                        result_tuple.append(tuple(tmp_tuple))
                    else:
                        if each_one is not None:
                            result_tuple.append(var(each_one.data.repeat(1, beam_size, 1)))
                        else:
                            result_tuple.append(None)
                return tuple(result_tuple)
            else: 
                return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', data_type)
        src_lengths = None

        self.tt = torch.cuda if self.cuda else torch
        src_lengths = self.tt.LongTensor(batch.src.size()[1]).fill_(batch.src.size()[0])
        # assert 
        # assert src_hist is not None, (batch.__dict__)

        emb = src

        total_n_corr_col = 0.0
        total_n_ex_col = 0
        total_dld_col = 0.0

        total_n_corr_row = 0.0
        total_n_ex_row = 0
        total_dld_row = 0.0

        if not stage1:
            enc_states, memory_bank, col_reps, row_reps, entity_reps = self.model2.encoder(emb, src_lengths)

            if self.is_num_ranking:
                sorted_list, indices_list, argmaxs = self.model2.col_sorted(col_reps, batch)

                n_corr_col, n_ex_col, dld_col = self.cal_pointer_metrics(argmaxs, indices_list)
                total_n_corr_col += n_corr_col.cpu().tolist()
                total_n_ex_col += n_ex_col
                total_dld_col += dld_col

            if self.is_imp_ranking:
                sorted_list, indices_list, argmaxs = self.model2.row_sorted(row_reps, batch)
                n_corr_row, n_ex_row, dld_row = self.cal_pointer_metrics(argmaxs, indices_list)
                total_n_corr_row += n_corr_row.cpu().tolist()
                total_n_ex_row += n_ex_row
                total_dld_row += dld_row

            model = self.model2

        dec_states = model.decoder.init_decoder_state(
                                        src, memory_bank, enc_states)


        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None

        memory_bank = rvar(memory_bank if isinstance(memory_bank, tuple) else memory_bank.data)
        memory_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)
            # Run one step.
            dec_out, dec_states, attn = model.decoder(
                inp, memory_bank, dec_states, memory_lengths=memory_lengths)

            if not stage1:
                dec_out = dec_out.squeeze(0)

            # (b) Compute a vector of batch x beam word scores.
            if not self.copy_attn:
                if stage1:
                    upd_attn = unbottle(attn["std"]).data
                    out = upd_attn
                else:
                    out = model.generator.forward(dec_out).data
                    out = unbottle(out)
                    # beam x tgt_vocab
                    beam_attn = unbottle(attn["std"])
            else:
                out = model.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map)

                # beam x (tgt_vocab + extra_vocab)

                out = data.collapse_copy_scores(
                    unbottle(out[0].data),
                    batch, self.fields[tgt].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()
                beam_attn = unbottle(attn["copy"])
            # (c) Advance each beam.
            for j, b in enumerate(beam):
                if stage1:
                    b.advance(
                        out[:, j],
                        torch.exp(unbottle(attn["std"]).data[:, j, :memory_lengths[j]]))
                else:
                    b.advance(out[:, j],
                        beam_attn.data[:, j, :memory_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)
        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size

        #if "tgt" in batch.__dict__:
        #    ret["gold_score"] = self._run_target(batch, datasets, indexes, unbottle)
        ret["batch"] = batch

        return ret, [total_n_corr_col, total_n_ex_col, total_dld_col], [total_n_corr_row, total_n_ex_row, total_dld_row]

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)

            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = onmt.io.make_features(batch, 'src', data_type)
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = \
            self.model.decoder.init_decoder_state(src, memory_bank, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        dec_out, dec_states, attn = self.model.decoder(
            tgt_in, memory_bank, dec_states, memory_lengths=src_lengths)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores

    def cal_pointer_metrics(self, argmaxs, indices):
        cnt = 0
        total = 0
        total_dld_scores = 0.0
        for i in range(len(argmaxs)):
            argmaxs[i] = argmaxs[i].squeeze(2)

            assert argmaxs[i].size() == indices[i].size()
            correct_cnt = argmaxs[i] == indices[i]

            dld_score = self.cal_dld(argmaxs[i], indices[i])
            total_dld_scores += dld_score
            total += correct_cnt.size(0) * correct_cnt.size(1)
            cnt += correct_cnt.sum()

        total_dld_scores = total_dld_scores / len(argmaxs)

        return cnt, total, total_dld_scores

    @staticmethod
    def cal_dld(s1, s2):
        """

        :param s1: n x len
        :param s2: n x len
        :return:
        """
        n_example = s1.size(0)
        sample = s1.view(-1).cpu().tolist()
        gold = s2.view(-1).cpu().tolist()
        total_scores = 1 - normalized_damerau_levenshtein_distance(sample, gold)

        return total_scores
