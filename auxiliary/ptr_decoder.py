import torch
import torch.nn as nn


class PtrAttention(nn.Module):
    def __init__(self, enc_hsz, dec_hsz, att_type='mlp'):
        super(PtrAttention, self).__init__()
        self.linear_query = nn.Linear(in_features=dec_hsz, out_features=dec_hsz, bias=True)
        self.linear_context = nn.Linear(in_features=enc_hsz, out_features=dec_hsz, bias=False)

        self.v = nn.Linear(in_features=dec_hsz, out_features=1, bias=False)
        self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def score(self, h_t, s):
        query = self.linear_query(h_t).unsqueeze(1)  # (bsz, 1, dec_hsz)
        context = self.linear_context(s)  # (bsz, enc_len, dec_hsz)
        value = self.tanh(query + context)
        att_dist = self.v(value).transpose(1, 2)
        return att_dist

    def forward(self, h_t, s, s_lens=None, mask=None):
        """

        :param h_t: (bsz, tgt_len, dec_hsz)
        :param s: (bsz, enc_len, enc_hsz)
        :param s_lens:
        :param mask
        :return:
        """

        att_dist = self.score(h_t=h_t, s=s)
        if s_lens is not None:
            assert False, s_lens

        if mask is not None:
            att_dist = att_dist.masked_fill_(mask.unsqueeze(1) == 1, -float('inf'))

        # att_dist = self.softmax(att_dist)
        att_dist = self.log_softmax(att_dist)
        # (bsz, 1, enc_len) x  (bsz, enc_len, enc_hsz)
        # context = torch.bmm(att_dist, s).squeeze(1)

        return att_dist, None


class PtrDecoder(nn.Module):
    def __init__(self, emb_dim, hsz, rnn_type='lstm', is_input_feed=False, dropout=0.3, att_type='mlp', sort_type='col',
                 is_mask=False):
        super(PtrDecoder, self).__init__()

        rnn_input_size = emb_dim + hsz if is_input_feed else emb_dim
        if rnn_type == 'gru':
            self.rnn = nn.GRUCell(input_size=rnn_input_size, hidden_size=hsz)
        else:
            self.rnn = nn.LSTMCell(input_size=rnn_input_size, hidden_size=hsz)

        self.attention = PtrAttention(enc_hsz=emb_dim, dec_hsz=hsz, att_type=att_type)
        self.rnn_type = rnn_type
        self.is_input_feed = is_input_feed
        self.dropout = nn.Dropout(p=dropout)
        self.init_embed = nn.Parameter(torch.FloatTensor(emb_dim), requires_grad=True)
        self.sort_type = sort_type
        self.is_mask = is_mask
        nn.init.uniform_(self.init_embed, -1, 1)

    def forward(self, home_col_rep, home_indices, vis_col_rep,
                vis_indices, team_col_rep, team_indices):
        home_sorted, home_argmax= self.once(home_col_rep, home_indices)
        vis_sorted, vis_argmax = self.once(vis_col_rep, vis_indices)
        team_sorted, team_argmax = self.once(team_col_rep, team_indices)
        return home_sorted, vis_sorted, team_sorted, [home_argmax, vis_argmax, team_argmax]

    def once(self, enc_outputs, indices, enc_lens=None):
        """
        :param enc_outputs: (kw_num, bsz, ply_num, hsz)
        :param indices: (bsz*kw_num, ply_num) or (bsz*ply_num, kw_num)
        :param enc_lens:
        :return:
        """
        assert self.sort_type in ['col', 'row', 'entity'], self.sort_type
        # assert enc_outputs.dim() == 4, enc_outputs.size()
        if self.sort_type == 'col':
            assert enc_outputs.size(2) == indices.size(1)
        elif self.sort_type == 'row':
            assert enc_outputs.size(0) == indices.size(1)

        if self.sort_type in ['col', 'row']:
            kw_num, bsz, ply_num, hsz = enc_outputs.size()
            if self.sort_type == 'col':
                # (bsz*kw_num, ply_num, hsz)
                enc_outputs = enc_outputs.transpose(0, 1).contiguous().view(-1, ply_num, hsz)
            else:
                # (kw_num, bsz, ply_num, hsz) -> (bsz * ply_num, kw_num, hsz)
                enc_outputs = enc_outputs.view(kw_num, -1, hsz).transpose(0, 1)

            new_bsz = bsz * kw_num if self.sort_type == 'col' else bsz * ply_num
        else:
            enc_outputs = enc_outputs.transpose(0, 1)  # bsz, n_entity, hsz
            new_bsz, _, hsz = enc_outputs.size()

        dec_inputs = torch.gather(enc_outputs, dim=1, index=indices.unsqueeze(2).repeat(1, 1, hsz))

        init_embed = self.init_embed.unsqueeze(0).unsqueeze(0).repeat(new_bsz, 1, 1)

        tgt = torch.cat([init_embed, dec_inputs], dim=1).transpose(0, 1)

        init_hidden, input_feed = self.init_decoder(enc_outputs)

        attns, pointer_argmaxs = self.decoding(enc_hs=enc_outputs, tgt=tgt[:-1], hidden=init_hidden,
                                               input_feed=input_feed, enc_lens=enc_lens, indices=indices)
        return attns, pointer_argmaxs

    def decoding(self, enc_hs, tgt, hidden, input_feed=None, enc_lens=None, indices=None):
        """

                :param enc_hs:
                :param tgt: (bsz) or (tgt_len, bsz)
                :param hidden:
                :param input_feed:
                :param enc_lens:
                :return:
                """
        assert tgt.dim() == 3
        attns = []
        pointer_argmaxs = []
        max_seq_len = enc_hs.size(1)
        mask = None

        if self.is_mask:
            mask = torch.zeros(enc_hs.size(0), max_seq_len, dtype=torch.int, requires_grad=False, device=enc_hs.device)
        for t in range(max_seq_len):
            emb_t = tgt[t]

            if self.is_input_feed:
                decoder_input = torch.cat([emb_t, input_feed], 1)
            else:
                decoder_input = emb_t

            if self.rnn_type == 'gru':
                hidden = self.rnn(decoder_input, hidden)
                rnn_output = hidden
            else:
                hidden = self.rnn(decoder_input, hidden)
                rnn_output = hidden[0]

            # assert False
            p_attn, decoder_output = self.attention(rnn_output, enc_hs, enc_lens, mask=mask)

            _, max_index = p_attn.max(dim=-1, keepdim=True)
            pointer_argmaxs.append(max_index)

            if self.is_input_feed:
                input_feed = decoder_output
            attns.append(p_attn)

            if self.is_mask:
                mask = mask.scatter_(1, indices[:, t].unsqueeze(1), 1)

        attns = torch.cat(attns, 1)
        pointer_argmaxs = torch.cat(pointer_argmaxs, 1)
        return attns, pointer_argmaxs

    def init_decoder(self, enc_outputs):
        enc_sum = torch.sum(enc_outputs, dim=1)
        if self.rnn_type == 'gru':
            init_hidden = enc_sum
        else:
            init_hidden = (enc_sum, enc_sum)

        input_feed = None
        if self.is_input_feed:
            bsz, enc_hsz = enc_sum.size()
            input_feed = enc_sum.new_zeros(bsz, enc_hsz)

        return init_hidden, input_feed


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, decoder_state, encoder_outputs, mask=None):
        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)

        # (batch_size, 1 (unsqueezed), hidden_size)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)

        # 1st line of Eq.(3) in the paper
        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)

        # softmax with only valid inputs, excluding zero padded parts
        # log-softmax for a better numerical stability
        log_score = self.softmax(u_i)

        return log_score


class PointerNet(nn.Module):
    def __init__(self, emb_dim, hsz, rnn_type='lstm', is_input_feed=False, dropout=0.3, att_type='mlp'):
        super(PointerNet, self).__init__()
        # emb_dim, hsz, rnn_type='lstm', is_input_feed=False, dropout=0.3, att_type='mlp'
        # Embedding dimension
        self.embedding_dim = emb_dim
        # (Decoder) hidden size
        self.hidden_size = hsz

        self.decoding_rnn = nn.LSTMCell(input_size=hsz, hidden_size=hsz)
        self.attn = Attention(hidden_size=hsz)

        self.rnn_type = rnn_type
        self.is_input_feed = is_input_feed

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, home_col_rep, vis_col_rep, team_col_rep):
        home_scores, home_pointer_argmaxs = self.decoding(home_col_rep)
        vis_scores, vis_pointer_argmaxs = self.decoding(vis_col_rep)
        team_scores, team_pointer_argmaxs = self.decoding(team_col_rep)

        return home_scores, home_pointer_argmaxs, vis_scores, vis_pointer_argmaxs, team_scores, team_pointer_argmaxs

    def decoding(self, encoder_outputs):
        pointer_scores = []
        pointer_argmaxs = []
        max_seq_len, bsz, _, hsz = encoder_outputs.size()

        encoder_outputs = encoder_outputs.view(-1, max_seq_len, hsz)

        new_bsz = encoder_outputs.size(0)
        hidden, input_feed = self.init_decoder(encoder_outputs)
        decoder_input = encoder_outputs.new_zeros(torch.Size((new_bsz, self.hidden_size)))

        for i in range(max_seq_len):
            # We will simply mask out when calculating attention or max (and loss later)
            # not all input and hiddens, just for simplicity
            # h, c: (batch_size, hidden_size)
            h_i, c_i = self.decoding_rnn(decoder_input, hidden)

            # next hidden
            hidden = (h_i, c_i)

            # Get a pointer distribution over the encoder outputs using attention
            # (batch_size, max_seq_len)
            pointer_score = self.attn(h_i, encoder_outputs)
            pointer_scores.append(pointer_score)

            # Get the indices of maximum pointer
            _, max_index = pointer_score.max(dim=-1, keepdim=True)
            pointer_argmaxs.append(max_index)
            index_tensor = max_index.unsqueeze(-1).expand(new_bsz, 1, self.hidden_size)

            # (batch_size, hidden_size)
            decoder_input = torch.gather(encoder_outputs, dim=1, index=index_tensor).squeeze(1)

        pointer_scores = torch.stack(pointer_scores, 1)
        pointer_argmaxs = torch.cat(pointer_argmaxs, 1)
        return pointer_scores, pointer_argmaxs

    def init_decoder(self, enc_outputs):

        enc_sum = torch.sum(enc_outputs, dim=1)
        if self.rnn_type == 'gru':
            init_hidden = enc_sum
        else:
            init_hidden = (enc_sum, enc_sum)

        input_feed = None
        if self.is_input_feed:
            bsz, enc_hsz = enc_sum.size()
            input_feed = enc_sum.new_zeros(enc_hsz).unsqueeze(0)

        return init_hidden, input_feed
