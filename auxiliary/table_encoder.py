import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.Models import EncoderBase
from onmt.modules.GlobalSelfAttention import GlobalSelfAttention
from auxiliary.multi_attention import MultiHeadAttention
from auxiliary.gnn import GCN, GAT, HGAT, GateGAT


class TableEncoderBlock(nn.Module):
    def __init__(self, emb_size, hidden_size, attn_type, attn_hidden, no_fusion, meta, two_dim_score=None, n_heads=8,
                 dropout=0.3, is_fnn=False, is_layer_norm=False, row_aggregation='mean', aggr_att_type='mlp',
                 row_agg_is_param=False):
        super(TableEncoderBlock, self).__init__()

        self.two_dim_score = two_dim_score
        self.meta = meta
        self.no_fusion = no_fusion
        self.is_fnn = is_fnn
        if attn_type == 'mul_head':
            self.row_rnn = MultiHeadAttention(n_heads=n_heads, dim=emb_size, dropout=dropout)
            self.col_attn = MultiHeadAttention(n_heads=n_heads, dim=emb_size, dropout=dropout)
        else:
            self.row_rnn = GlobalSelfAttention(emb_size, coverage=False, attn_type=attn_type,
                                               attn_hidden=attn_hidden, no_gate=True, no_gate_bias=True,
                                               no_fusion=no_fusion)
            self.col_attn = GlobalSelfAttention(emb_size, coverage=False, attn_type=attn_type,
                                                attn_hidden=attn_hidden, no_gate=True, no_gate_bias=True,
                                                no_fusion=no_fusion)
        self.two_dim_gen_layer = nn.Sequential()
        two_dim = True
        only_concated = False

        two_dim_gen_layer_input_dim = hidden_size * 2
        self.two_dim = two_dim

        # if two_dim:
        #     self.two_fusion = nn.Sequential()
        #     self.two_fusion.add_module("linear_transform", nn.Linear(hidden_size * 2, hidden_size))

        #    # self.two_fusion.add_module("tanh", nn.Tanh())
        self.two_dim_gen_layer.add_module('linear_transform',
                                          nn.Linear(two_dim_gen_layer_input_dim, hidden_size, bias=False))

        self.two_dim_gen_layer.add_module('tanh', nn.Tanh())
        if not only_concated:
            assert two_dim_score in ["mlp", "general", "dot"]

            if two_dim_score == "mlp":
                self.two_dim_mlp_score_layer = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size, bias=False),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1, bias=False)
                )
            elif two_dim_score == "general":
                self.two_dim_general_score_layer = nn.Linear(hidden_size, hidden_size, bias=False)

        if row_aggregation == 'attention':
            self.row_aggregator = RowAggregation(hsz=hidden_size, attn_type=aggr_att_type, is_param=row_agg_is_param)
        self.row_aggregation = row_aggregation
        self.only_concated = only_concated

    def col_dim_encoding(self, memory_bank, batch_size, memory_lengths):
        # size is (ply_num, batch, kw_num, hidden)
        col_memory_bank = memory_bank.contiguous().view(memory_bank.size(0), batch_size, -1,
                                                        memory_bank.size(2)).transpose(0, 2).contiguous()

        # size is (ply_num, batch*kw_num, hidden)
        col_memory_bank = col_memory_bank.view(col_memory_bank.size(0), -1, col_memory_bank.size(3))

        # size is (ply_num, batch*kw_num, dim)
        col_rep, _, col_context = self.col_attn(col_memory_bank.transpose(0, 1).contiguous(),
                                                col_memory_bank.transpose(0, 1),
                                                memory_lengths=memory_lengths.expand(col_memory_bank.size(1)))

        assert col_rep.size() == col_context.size(), "{}: {}".format(col_rep.size(), col_context.size())
        # (kw_num, bsz, ply_num, hsz)
        col_rep = col_rep.view(col_rep.size(0), batch_size, -1, col_rep.size(2)).transpose(0, 2).contiguous()
        col_context = col_context.view(col_context.size(0), batch_size, -1, col_context.size(2)).transpose(0, 2).contiguous()
        # size is (kw_num, batch*ply_num, dim)
        col_rep = col_rep.view(col_rep.size(0), -1, col_rep.size(3))
        col_context = col_context.view(col_context.size(0), -1, col_context.size(3))
        return col_rep, col_context

    def forward(self, emb, start_id, end_id, num, num_tensor, kw_num, kw_num_tensor, batch):
        # (ply_num*kw_num, bsz, hsz)
        home_ply_rep = emb.narrow(0, start_id, (end_id - start_id + 1))
        # size are (kw_num, batch*ply_num, hidden)
        home_ply_rep = home_ply_rep.transpose(0, 1).contiguous().view(home_ply_rep.size(1) * int(num), kw_num,
                                                                      home_ply_rep.size(2)).transpose(0, 1)

        # col dimension encoding
        # (kw_num, batch*ply_num, dim)
        col_dim_rep, col_context = self.col_dim_encoding(home_ply_rep, batch, num_tensor)
        row_encoding_input = col_dim_rep
        # row_encoding_input = home_ply_rep
        # row dimension encoding
        # (kw_num, batch*ply_num, dim)
        row_dim_rep, _, row_context = self.row_rnn(row_encoding_input.transpose(0, 1).contiguous(),
                                                   row_encoding_input.transpose(0, 1),
                                                   memory_lengths=kw_num_tensor.expand(home_ply_rep.size(1)))
        hsz = row_dim_rep.size(2)

        cated_features = torch.cat((row_dim_rep, col_dim_rep), dim=2)
        gen_rep = self.two_dim_gen_layer(cated_features)

        row_dim1, row_dim2, row_dim3 = row_dim_rep.size()
        if not self.only_concated:
            if self.two_dim_score == "mlp":
                row_score = self.two_dim_mlp_score_layer(torch.cat((row_dim_rep, gen_rep), 2))
                col_score = self.two_dim_mlp_score_layer(torch.cat((col_dim_rep, gen_rep), 2))

            elif self.two_dim_score == "general":
                row_score = torch.bmm(
                    row_dim_rep.view(-1, row_dim3).unsqueeze(1),
                    self.two_dim_general_score_layer(gen_rep).view(-1, row_dim3).unsqueeze(2)
                ).view(row_dim1, row_dim2, 1)
                col_score = torch.bmm(
                    col_dim_rep.view(-1, row_dim3).unsqueeze(1),
                    self.two_dim_general_score_layer(gen_rep).view(-1, row_dim3).unsqueeze(2)
                ).view(row_dim1, row_dim2, 1)

            elif self.two_dim_score == "dot":
                row_score = torch.bmm(
                    row_dim_rep.view(-1, row_dim3).unsqueeze(1),
                    gen_rep.view(-1, row_dim3).unsqueeze(2)
                ).view(row_dim1, row_dim2, 1)
                col_score = torch.bmm(
                    col_dim_rep.view(-1, row_dim3).unsqueeze(1),
                    gen_rep.view(-1, row_dim3).unsqueeze(2)
                ).view(row_dim1, row_dim2, 1)
            else:
                assert False, self.two_dim_score

            two_dim_weight = F.softmax(torch.cat((row_score, col_score), 2), dim=2)

            rep_concat = torch.cat((row_dim_rep.unsqueeze(2), col_dim_rep.unsqueeze(2)), 2)
            assert rep_concat.size(2) == 2
            assert two_dim_weight.size(2) == 2

            num_cated = 2

            rep_view = rep_concat.view(-1, num_cated, row_dim3)
            score_view = two_dim_weight.view(-1, num_cated).unsqueeze(1)

            # (kw_num, batch*ply_num, hidden)
            home_ply_memory_bank = torch.bmm(score_view, rep_view).squeeze(1).view(row_dim1, row_dim2, row_dim3)
        else:
            home_ply_memory_bank = gen_rep

        if self.no_fusion:
            home_ply_memory_bank = home_ply_memory_bank + home_ply_rep

        # home_ply_memory_bank = self.two_fusion(torch.cat([home_ply_rep, home_ply_memory_bank], dim=2))
        # row_for_ptr = row_context.view(int(kw_num), -1, int(num), hsz)
        # col_for_ptr = col_context.view(int(kw_num), -1, int(num), hsz)
        row_for_ptr = row_dim_rep.view(int(kw_num), -1, int(num), hsz)
        col_for_ptr = col_dim_rep.view(int(kw_num), -1, int(num), hsz)

        if self.row_aggregation == 'mean':
            home_ply_row_rep = home_ply_memory_bank.mean(0)
        elif self.row_aggregation == 'attention':
            home_ply_row_rep = self.row_aggregator(home_ply_memory_bank)

        # (kw_num*ply_num, bsz, hsz)
        home_ply_row_rep = home_ply_row_rep.view(batch, -1, home_ply_memory_bank.size(2)).\
            transpose(0, 1).contiguous()
        # changing shape for ptr decoding

        # row_dim_rep = row_dim_rep.view(int(kw_num), -1, int(num), hsz)
        # col_dim_rep = col_dim_rep.view(int(kw_num), -1, int(num), hsz)

        return home_ply_row_rep, home_ply_memory_bank, col_for_ptr, row_for_ptr

    def only_col_dimension(self, emb, start_id, end_id, num, num_tensor, kw_num, kw_num_tensor, batch):
        # (ply_num*kw_num, bsz, hsz)
        home_ply_rep = emb.narrow(0, start_id, (end_id - start_id + 1))
        # size are (kw_num, batch*ply_num, hidden)
        home_ply_rep = home_ply_rep.transpose(0, 1).contiguous().view(home_ply_rep.size(1) * int(num), kw_num,
                                                                      home_ply_rep.size(2)).transpose(0, 1)

        # col dimension encoding
        # (kw_num, batch*ply_num, dim)
        col_dim_rep, col_context = self.col_dim_encoding(home_ply_rep, batch, num_tensor)
        col_for_ptr = col_dim_rep.view(int(kw_num), -1, int(num), col_dim_rep.size(2))
        return col_for_ptr

    def only_col_and_row(self, emb, start_id, end_id, num, num_tensor, kw_num, kw_num_tensor, batch):
        # (ply_num*kw_num, bsz, hsz)
        home_ply_rep = emb.narrow(0, start_id, (end_id - start_id + 1))
        # size are (kw_num, batch*ply_num, hidden)
        home_ply_rep = home_ply_rep.transpose(0, 1).contiguous().view(home_ply_rep.size(1) * int(num), kw_num,
                                                                      home_ply_rep.size(2)).transpose(0, 1)

        # col dimension encoding
        # (kw_num, batch*ply_num, dim)
        col_dim_rep, col_context = self.col_dim_encoding(home_ply_rep, batch, num_tensor)
        row_encoding_input = col_dim_rep
        # row dimension encoding
        # (kw_num, batch*ply_num, dim)
        row_dim_rep, _, row_context = self.row_rnn(row_encoding_input.transpose(0, 1).contiguous(),
                                                   row_encoding_input.transpose(0, 1),
                                                   memory_lengths=kw_num_tensor.expand(home_ply_rep.size(1)))
        hsz = row_dim_rep.size(2)

        row_for_ptr = row_dim_rep.view(int(kw_num), -1, int(num), hsz)
        col_for_ptr = col_dim_rep.view(int(kw_num), -1, int(num), hsz)

        return col_for_ptr, row_for_ptr


class TableEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, meta, num_layers, embeddings, emb_size, attn_hidden,
                 bidirectional, hidden_size, dropout=0.0, attn_type="general", table_attn_type='general',
                 two_dim_score=None,  no_fusion=False, n_heads=8,
                 is_fnn=False, is_layer_norm=False, row_aggregation='mean', aggr_att_type='mlp',
                 row_rep_type='attn',
                 gnn_type='gcn', gnn_share=False, gnn_n_layers=2,
                 gat_dropout=0.3, gat_is_self=False, gat_n_heads=1, gnn_activation_type='relu', player_player=True,
                 row_agg_is_param=False, gat_attn_type='dot', residual=False, leaky_relu_alpha=0.2,
                 using_global_node=False):
        super(TableEncoder, self).__init__()

        self.table_encoding_block = TableEncoderBlock(emb_size=emb_size, hidden_size=hidden_size,
                                                      attn_type=table_attn_type,
                                                      attn_hidden=attn_hidden, no_fusion=no_fusion, meta=meta,
                                                      two_dim_score=two_dim_score, n_heads=n_heads, dropout=dropout,
                                                      is_fnn=is_fnn, is_layer_norm=is_layer_norm,
                                                      row_aggregation=row_aggregation, aggr_att_type=aggr_att_type,
                                                      row_agg_is_param=row_agg_is_param)

        if row_rep_type == 'gnn':
            using_gnn = True
        else:
            using_gnn = False

        if row_rep_type == 'gnn':

            if gnn_type == 'gcn':
                self.gnn = GCN(dim_in=hidden_size, dim_out=hidden_size, n_layers=gnn_n_layers, is_sharing=gnn_share,
                               is_self=gat_is_self, player_player=player_player)
            elif gnn_type == 'gat':
                self.gnn = GAT(hidden_size, hidden_size, dropout=gat_dropout, n_layers=gnn_n_layers,
                               is_sharing=gnn_share, is_self=gat_is_self, n_heads=gat_n_heads,
                               activation_type=gnn_activation_type, player_player=player_player,
                               gat_attn_type=gat_attn_type,
                               residual=residual, leaky_relu_alpha=leaky_relu_alpha)

            elif gnn_type == 'gategat':
                self.gnn = GateGAT(in_features=hidden_size,
                                   out_features=hidden_size if gat_attn_type == 'dot' else attn_hidden,
                                   dropout=gat_dropout,
                                   n_layers=gnn_n_layers, is_sharing=gnn_share, is_self=gat_is_self,
                                   n_heads=gat_n_heads, player_player=player_player, gat_attn_type=gat_attn_type,
                                   using_global_node=using_global_node,
                                   activate=gnn_activation_type)
            elif gnn_type == 'hgat':
                self.gnn = HGAT(in_features=hidden_size, out_features=hidden_size,
                                dropout=gat_dropout, n_heads=gat_n_heads, n_layers=gnn_n_layers, is_self=gat_is_self,
                                is_sharing=gnn_share, gnn_activation_type=gnn_activation_type,
                                player_player=player_player)

            self.using_global_node = using_global_node
        elif row_rep_type == 'attn':
            self.row_attn = GlobalSelfAttention(hidden_size, coverage=False, attn_type=attn_type,
                                                attn_hidden=attn_hidden, no_gate=False, no_gate_bias=True)
        self.using_gnn = using_gnn
        self.row_rep_type = row_rep_type

        self.num_layers = num_layers

        if isinstance(embeddings, tuple):
            self.embeddings = embeddings[0]
            self.hist_embeddings = embeddings[1]
        else:
            self.embeddings = embeddings
            self.hist_embeddings = None

        self.meta = meta
        self.dropout = nn.Dropout(p=dropout)
        bidirectional = bidirectional if bidirectional is not None else False
        self.bidirectional = bidirectional

        self.hidden_size = hidden_size

        assert hidden_size == emb_size
        # col dim record
        # self.col_sigmoid = self.two_dim_mlp = None
        self.no_fusion = no_fusion
        # self.col_sigmoid = nn.Sequential(
        #     nn.Linear(hidden_size + emb_size, hidden_size),
        #     nn.Sigmoid()
        # )

        self.register_buffer("row_attn_length", torch.LongTensor(
            [int(self.meta['home_ply_num']) + int(self.meta['vis_ply_num']) + int(self.meta['team_num'])]))
        self.register_buffer("home_ply_length", torch.LongTensor([int(self.meta['home_ply_num'])]))
        self.register_buffer("vis_ply_length", torch.LongTensor([int(self.meta['vis_ply_num'])]))
        self.register_buffer("team_length", torch.LongTensor([int(self.meta['team_num'])]))

        self.register_buffer("home_ply_num_kw_num", torch.LongTensor([int(self.meta["home_ply_kw_num"])]))
        self.register_buffer("vis_ply_num_kw_num", torch.LongTensor([int(self.meta["vis_ply_kw_num"])]))
        self.register_buffer("team_kw_num", torch.LongTensor([int(self.meta["team_kw_num"])]))

    def obtainLastLayer(self, rep, batch_size, hier_hist_seq=False):
        if hier_hist_seq:
            length = 1
        else:
            length = 2 if self.bidirectional else 1
        rep = rep.narrow(0, rep.size(0) - length, length).transpose(0, 1).contiguous().view(rep.size(1), -1)
        rep = rep.view(batch_size, -1, rep.size(1)).transpose(0, 1).contiguous()
        return rep

    # both rnn_output, memory_bank size are (kw_num, batch*ply_num, hidden)
    # rnn_output is the representation of row_dim encoding

    def forward(self, src, lengths=None, encoder_state=None, memory_lengths=None):
        "See :obj:`EncoderBase.forward()`"
        # if isinstance(src, tuple) and isinstance(src[0], tuple) and len(src) == 2:
        #     src, hist_src = src
        # else:
        #      hist_src = None
        self._check_args(src, lengths, encoder_state)

        # (kw_num*play_num=606, bsz, 4) -> (kw_num*play_num, bsz, emb_dim)
        token_embedded = self.embeddings(src)
        s_len, batch, emb_dim = token_embedded.size()

        emb = self.dropout(token_embedded)
        # obtain record level representation
        assert s_len == self.meta['tuple_num'], "{}: {}".format(s_len, self.meta['tuple_num'])

        # (4, bsz, emb_dim)
        special_rep = emb.narrow(0, self.meta['special_start'],
                                 (self.meta['special_end'] - self.meta['special_start'] + 1))

        home_ply_row_rep, home_ply_memory_bank, \
        home_col_dim_rep, home_row_dim_rep = self.table_encoding_block(emb, self.meta['home_ply_start'],
                                                                       self.meta['home_ply_end'],
                                                                       self.meta['home_ply_num'],
                                                                       self.home_ply_length,
                                                                       self.meta['home_ply_kw_num'],
                                                                       self.home_ply_num_kw_num, batch)

        vis_ply_row_rep, vis_ply_memory_bank, \
        vis_col_dim_rep, vis_row_dim_rep = self.table_encoding_block(emb, self.meta['vis_ply_start'],
                                                                     self.meta['vis_ply_end'], self.meta['vis_ply_num'],
                                                                     self.vis_ply_length, self.meta['vis_ply_kw_num'],
                                                                     self.vis_ply_num_kw_num, batch)

        team_row_rep, team_memory_bank, \
        team_col_dim_rep, team_row_dim_rep = self.table_encoding_block(emb, self.meta['team_start'], self.meta['team_end'],
                                                                       self.meta['team_num'], self.team_length,
                                                                       self.meta['team_kw_num'], self.team_kw_num, batch)

        total_row_rep = torch.cat((home_ply_row_rep, vis_ply_row_rep, team_row_rep), 0)

        if self.row_rep_type == 'gnn':
            if self.using_global_node:
                row_rep_mean = torch.sum(total_row_rep, dim=0, keepdim=True)
                total_row_rep = torch.cat([total_row_rep, row_rep_mean], dim=0)
            total_row_rep = self.gnn(total_row_rep.transpose(0, 1).contiguous())

        elif self.row_rep_type == 'attn':
            total_row_rep, _, _ = self.row_attn(total_row_rep.transpose(0, 1).contiguous(), total_row_rep.transpose(0, 1),
                                                memory_lengths=self.row_attn_length.expand(batch))

        # vectors_vision(total_row_rep[:, 0, :].cpu())
        # assert False
        home_ply_row_rep = total_row_rep.narrow(0, 0, home_ply_row_rep.size(0))
        vis_ply_row_rep = total_row_rep.narrow(0, home_ply_row_rep.size(0), vis_ply_row_rep.size(0))
        team_row_rep = total_row_rep.narrow(0, home_ply_row_rep.size(0) + vis_ply_row_rep.size(0), team_row_rep.size(0))
        if self.row_rep_type == 'gnn' and self.using_global_node:
            mean = total_row_rep.narrow(0, home_ply_row_rep.size(0) + vis_ply_row_rep.size(0) + team_row_rep.size(0), 1)
            mean = mean.expand(self.num_layers, batch, self.hidden_size)
        else:
            mean = total_row_rep.mean(0).expand(self.num_layers, batch, self.hidden_size)

        mem_bank_as_orig = torch.cat((special_rep,
                                      torch.cat(
                                          [tmp_i.transpose(0, 1).contiguous().view(batch, -1, tmp_i.size(2)).transpose(0, 1)
                                                 for tmp_i in (home_ply_memory_bank, vis_ply_memory_bank, team_memory_bank)],
                                          0)
                                      ), 0)

        memory_bank = ((special_rep, home_ply_memory_bank, vis_ply_memory_bank, team_memory_bank),
                       (special_rep, home_ply_row_rep, vis_ply_row_rep, team_row_rep), mem_bank_as_orig)

        encoder_final = (mean, mean)
        # home_rep = home_ply_memory_bank.view(home_row_dim_rep.size())
        # vis_rep = vis_ply_memory_bank.view(vis_row_dim_rep.size())
        # team_rep = team_memory_bank.view(team_row_dim_rep.size())

        return encoder_final, memory_bank, (home_col_dim_rep, vis_col_dim_rep, team_col_dim_rep), \
               (home_row_dim_rep, vis_row_dim_rep, team_row_dim_rep), total_row_rep
        # return encoder_final, memory_bank, (home_rep, vis_rep, team_rep), (home_rep, vis_rep, team_rep)


    def only_col_and_row_encoding(self, src, lengths=None, encoder_state=None):
        if isinstance(src, tuple) and isinstance(src[0], tuple) and len(src) == 2:
            src, hist_src = src
        else:
            hist_src = None

        self._check_args(src, lengths, encoder_state)
        assert hist_src is None

        # (kw_num*play_num=606, bsz, 4) -> (kw_num*play_num, bsz, emb_dim)
        token_embedded = self.embeddings(src)
        s_len, batch, emb_dim = token_embedded.size()
        emb = self.dropout(token_embedded)

        home_col_dim_rep, home_row_dim_rep = self.table_encoding_block.only_col_and_row(emb, self.meta['home_ply_start'],
                                                                                        self.meta['home_ply_end'],
                                                                                        self.meta['home_ply_num'],
                                                                                        self.home_ply_length,
                                                                                        self.meta['home_ply_kw_num'],
                                                                                        self.home_ply_num_kw_num, batch)
        vis_col_dim_rep, vis_row_dim_rep = self.table_encoding_block.only_col_and_row(emb, self.meta['vis_ply_start'],
                                                                                      self.meta['vis_ply_end'],
                                                                                      self.meta['vis_ply_num'],
                                                                                      self.vis_ply_length,
                                                                                      self.meta['vis_ply_kw_num'],
                                                                                      self.vis_ply_num_kw_num, batch)

        team_col_dim_rep, team_row_dim_rep = self.table_encoding_block.only_col_and_row(emb, self.meta['team_start'],
                                                                                        self.meta['team_end'],
                                                                                        self.meta['team_num'], self.team_length,
                                                                                        self.meta['team_kw_num'],
                                                                                        self.team_kw_num, batch)

        return (home_col_dim_rep, vis_col_dim_rep, team_col_dim_rep),\
               (home_row_dim_rep, vis_row_dim_rep, team_row_dim_rep), None

    def all_ptr_encoding(self, src, lengths=None, encoder_state=None):
        if isinstance(src, tuple) and isinstance(src[0], tuple) and len(src) == 2:
            src, hist_src = src
        else:
            hist_src = None

        self._check_args(src, lengths, encoder_state)
        assert hist_src is None

        token_embedded = self.embeddings(src)
        s_len, batch, emb_dim = token_embedded.size()

        emb = self.dropout(token_embedded)
        # obtain record level representation
        assert s_len == self.meta['tuple_num'], "{}: {}".format(s_len, self.meta['tuple_num'])

        # (4, bsz, emb_dim)
        # special_rep = emb.narrow(0, self.meta['special_start'],
        #                          (self.meta['special_end'] - self.meta['special_start'] + 1))

        home_ply_row_rep, _, \
        home_col_dim_rep, home_row_dim_rep = self.table_encoding_block(emb, self.meta['home_ply_start'],
                                                                       self.meta['home_ply_end'],
                                                                       self.meta['home_ply_num'],
                                                                       self.home_ply_length,
                                                                       self.meta['home_ply_kw_num'],
                                                                       self.home_ply_num_kw_num, batch)

        vis_ply_row_rep, _, \
        vis_col_dim_rep, vis_row_dim_rep = self.table_encoding_block(emb, self.meta['vis_ply_start'],
                                                                     self.meta['vis_ply_end'], self.meta['vis_ply_num'],
                                                                     self.vis_ply_length, self.meta['vis_ply_kw_num'],
                                                                     self.vis_ply_num_kw_num, batch)

        team_row_rep, _, \
        team_col_dim_rep, team_row_dim_rep = self.table_encoding_block(emb, self.meta['team_start'],
                                                                       self.meta['team_end'],
                                                                       self.meta['team_num'], self.team_length,
                                                                       self.meta['team_kw_num'], self.team_kw_num,
                                                                       batch)

        total_row_rep = torch.cat((home_ply_row_rep, vis_ply_row_rep, team_row_rep), 0)

        if self.row_rep_type == 'gnn':
            if self.using_global_node:
                row_rep_mean = torch.sum(total_row_rep, dim=0, keepdim=True)
                total_row_rep = torch.cat([total_row_rep, row_rep_mean], dim=0)
            total_row_rep = self.gnn(total_row_rep.transpose(0, 1).contiguous())

        elif self.row_rep_type == 'attn':
            total_row_rep, _, _ = self.row_attn(total_row_rep.transpose(0, 1).contiguous(),
                                                total_row_rep.transpose(0, 1),
                                                memory_lengths=self.row_attn_length.expand(batch))

        return (home_col_dim_rep, vis_col_dim_rep, team_col_dim_rep),\
               (home_row_dim_rep, vis_row_dim_rep, team_row_dim_rep), total_row_rep


class RowAggregation(nn.Module):
    def __init__(self, hsz, attn_type='mlp', is_param=False):
        super(RowAggregation, self).__init__()

        if attn_type == 'mlp':
            self.mlp_score_layer = nn.Sequential(
                nn.Linear(in_features=hsz*2, out_features=hsz, bias=False),
                nn.Tanh(),
                nn.Linear(in_features=hsz, out_features=1, bias=False)
            )
        elif attn_type == 'general':
            self.general_score_layer = nn.Linear(in_features=hsz, out_features=hsz, bias=False)
        elif attn_type == 'dot':
            self.param_w = nn.Linear(in_features=hsz, out_features=hsz, bias=True)
            self.param_tanh = nn.Tanh()

        self.attn_type = attn_type
        self.softmax = nn.Softmax(dim=1)
        # assert False
        if is_param:
            self.param = nn.Parameter(torch.FloatTensor(hsz))
            self.param.requires_grad = True

        self.is_param = is_param

    def forward(self, table_memory_bank):
        """

        :param table_memory_bank: (n_kw, n_row*bsz, hsz)
        :return:
        """
        n_kw, bsz, _ = table_memory_bank.size()
        if self.is_param:
            # param = self.param_tanh(self.param_w(self.param))
            mean = self.param.view(1, 1, -1).repeat(bsz, 1, 1)
        else:
            mean = torch.mean(table_memory_bank, dim=0, keepdim=True).transpose(0, 1)  # (n_row*bsz, 1, hsz)

        table_memory_bank = table_memory_bank.transpose(0, 1)  # (n_row*bsz, n_kw, hsz)

        if self.attn_type == 'mlp':
            repeated_mean = mean.repeat(1, n_kw, 1)
            cated_feature = torch.cat([repeated_mean, table_memory_bank], dim=2)
            score = self.mlp_score_layer(cated_feature)  # (n_row*bsz, n_kw, 1)
        elif self.attn_type == 'general':
            assert False, self.attn_type
        elif self.attn_type == 'dot':
            table_memory_bank_feats = self.param_tanh(self.param_w(table_memory_bank))
            # (n_row * bsz, n_kw, hsz) (n_row * bsz, 1, hsz) -> (n_row * bsz, 1, n_kw)
            score = torch.bmm(table_memory_bank_feats, mean.transpose(1, 2))
        else:
            assert False, self.attn_type
        weight = self.softmax(score)
        # (n_row*bsz, n_kw, 1) * (n_row*bsz, n_kw, hsz) > (n_row*bsz, 1, hsz)
        row_rep = torch.bmm(weight.transpose(1, 2), table_memory_bank).squeeze(1)
        return row_rep


def get_masks(slen, lengths):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    # assert lengths.max().item() <= slen
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    # mask = alen >= lengths[:, None]
    mask = alen < lengths[:, None]
    return mask