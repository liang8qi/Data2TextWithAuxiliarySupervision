import torch
import torch.nn as nn

import math
from auxiliary.utils import vectors_vision
from auxiliary.gelu import GELU


class GCNLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features=dim_in, out_features=dim_out, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        fc_output = self.fc(A.matmul(x))
        return self.relu(fc_output)


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, n_layers=1, is_sharing=False, is_self=True, player_player=True):
        super(GCN, self).__init__()

        A = get_adj_matrix(is_self=is_self, is_normed=True, player_player=player_player)
        self.A = nn.Parameter(A)
        self.A.requires_grad = False

        # if n_layers == 1:
        #     self.gcns = GCNLayer(dim_in, dim_out)
        # else:
        if is_sharing:
            gcn = GCNLayer(dim_in, dim_out)
            self.gcns = nn.Sequential()
            for i in range(n_layers):
                self.gcns.add_module("gcn-{}".format(i), gcn)
        else:
            self.gcns = nn.Sequential()
            for i in range(n_layers):
                self.gcns.add_module("gcn-{}".format(i), GCNLayer(dim_in, dim_out))

        self.n_layers = n_layers
        self.is_self = is_self

    def forward(self, x):
        """

        :param x:
        :return: bsz, n, hsz
        """
        bsz, n_entity, hsz = x.size()
        # x = x.transpose(0, 1)
        # print(self.A.size())
        A = self.A.unsqueeze(0).expand(bsz, n_entity, n_entity)
        for gcn_layer in self.gcns:
            out = gcn_layer(x, A)
            if self.is_self:
                x = out
            else:
                x = x + out

        return x.transpose(0, 1)


def get_adj_matrix(is_normed=False, symmetric=True, is_self=True, player_player=True, using_global_node=False):
    # 1-13: home player
    # 14-26 vis player
    # 27: home team
    # 28: vis team
    if player_player:
        home_player = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        vis_player = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    else:
        home_player = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        vis_player = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    home_team = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    vis_team = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    if using_global_node:
        home_player.append(0)
        vis_player.append(0)
        home_team.append(0)
        vis_team.append(0)

    matrix = []
    for i in range(13):
        matrix.append(home_player.copy())
    # print(matrix)
    for i in range(13):
        matrix.append(vis_player.copy())
    matrix.append(home_team)
    matrix.append(vis_team)
    if using_global_node:
        matrix.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    if not is_self:
        for i in range(len(matrix)):
            matrix[i][i] = 0

    A = torch.tensor(matrix, dtype=torch.float, requires_grad=False)

    if not is_normed:
        return A
    # print(A)
    d = A.sum(1)

    if symmetric:
        # D = D^-1/2
        D = torch.diag(torch.pow(d, -0.5))
        A_normed = D.mm(A).mm(D)
        # print(A_normed)
        # assert False
        return A_normed
    else:
        # D=D^-1
        D = torch.diag(torch.pow(d, -1))
        A_normed = D.mm(A)
        return A_normed


class SingleHeadGeneralGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha=0.2, activation_type='relu', concat=True):
        super(SingleHeadGeneralGATLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.W = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        self.a = nn.Linear(in_features=2*out_features, out_features=1, bias=False)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)

        self.softmax = nn.Softmax(dim=2)
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        elif activation_type == 'elu':
            self.activation = nn.ELU()
        else:
            assert False, activation_type

        self.concat = concat
        self.out_features = out_features

    def forward(self, x, adj):
        """
        :param x: [bsz, n_player+n_team, hsz]
        :param adj: [bsz, n_player+n_team, n_player+n_team]
        :return:
        """
        w_x = self.W(x)
        # bsz, n, n, 1
        a_input = self._prepare_attentional_mechanism_input(query=w_x, key=w_x)
        e = self.leaky_relu(self.a(a_input).squeeze(3))
        attention = e.masked_fill_(~adj.bool(), -float('inf'))  # [bsz, n, n]
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        outputs = torch.bmm(attention, w_x)
        if self.concat:
            outputs = self.activation(outputs)
        return outputs

    def _prepare_attentional_mechanism_input(self, query, key):
        bsz, n, hsz = query.size()

        wh_repeated_in_chunks = query.repeat_interleave(n, dim=1)
        wh_repeated_alternating = key.repeat(1, n, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (bsz, N * N, out_features)

        all_combinations_matrix = torch.cat([wh_repeated_in_chunks, wh_repeated_alternating], dim=2)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(bsz, n, n, 2 * self.out_features)


class GeneralGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, n_heads=1, alpha=0.2, activation_type='relu', residual=False):
        super(GeneralGATLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)

        d_k_in = in_features // n_heads
        d_k_out = out_features // n_heads

        self.n_head_attns = nn.ModuleList()
        for i in range(n_heads):
            self.n_head_attns.add_module('{}_head'.format(i),
                                         SingleHeadGeneralGATLayer(in_features=d_k_in, out_features=d_k_out,
                                                                   dropout=dropout, alpha=alpha,
                                                                   activation_type=activation_type,
                                                                   concat=True))
        self.n_heads = n_heads
        self.d_k_in = d_k_in
        # self.out_attn = SingleHeadGeneralGATLayer(in_features=out_features, out_features=out_features, dropout=dropout,
        #                                           alpha=alpha, concat=False)
        self.residual = residual

    def forward(self, x, adj):
        """
        :param x: [bsz, n_player+n_team, hsz]
        :param adj: [bsz, n_player+n_team, n_player+n_team]
        :return:
        """
        x = self.dropout(x)
        n_head_x = x.split(self.d_k_in, dim=2)

        outputs = torch.cat([one_head_attn(n_head_x[i], adj) for i, one_head_attn in enumerate(self.n_head_attns)], dim=2)
        outputs = self.dropout(outputs)
        return outputs


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, n_heads=1, activation_type='relu'):
        super(GATLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.key_projection = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.query_projection = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.value_projection = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.output = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        # self.leaky_relu = nn.LeakyReLU(self.alpha_leak_relu)
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        elif activation_type == 'elu':
            self.activation = nn.ELU()
        else:
            assert False, activation_type

        self.softmax = nn.Softmax(dim=2)

        self.dropout = nn.Dropout(p=dropout)

        self.d_k = out_features // n_heads

        self.sqrt_d_k = math.sqrt(self.d_k) if n_heads > 1 else 1
        self.n_heads = n_heads

    def forward(self, x, adj):
        """
        :param x: [bsz, n_player+n_team, hsz]
        :param adj: [bsz, n_player+n_team, n_player+n_team]
        :return:
        """
        bsz, n, hsz = x.size()
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        def shape_projection(tensor):
            b, l, d = tensor.size()
            return tensor.view(b, l, self.n_heads, self.d_k) \
                .transpose(1, 2).contiguous() \
                .view(b * self.n_heads, l, self.d_k)

        query_up = shape_projection(query)
        key_up = shape_projection(key)
        value_up = shape_projection(value)
        scaled = torch.bmm(query_up, key_up.transpose(1, 2))
        scaled = scaled / self.sqrt_d_k
        scaled = scaled.masked_fill_(~adj.bool(), -float('inf'))  # [bsz*n_heads, n, n]
        # print(scaled[0])
        attention = self.softmax(scaled)
        attention = self.dropout(attention)

        # [bsz, n_heads, n, n] [bsz, n_heads, n, d_k] -> [bsz, n_heads, n, d_k]
        outputs = torch.bmm(attention.view(-1, n, n), value_up)
        outputs = outputs.view(bsz, self.n_heads, n, self.d_k).\
            transpose(1, 2).contiguous().view(bsz, n, self.n_heads*self.d_k)
        outputs = self.output(self.activation(outputs))
        # outputs = self.activation(outputs)

        return outputs


class GAT(nn.Module):
    def __init__(self, in_features, out_features, dropout,
                 n_layers=2, is_sharing=False, is_self=False, n_heads=1, activation_type='relu', player_player=True,
                 gat_attn_type='dot', residual=False, leaky_relu_alpha=0.2):
        super(GAT, self).__init__()

        assert n_layers > 0
        assert is_self == True or residual == True
        assert gat_attn_type in ['dot', 'general']
        adj = get_adj_matrix(is_normed=False, is_self=is_self, player_player=player_player)
        self.adj = nn.Parameter(adj)
        self.adj.requires_grad = False
        self.gats = nn.Sequential()

        if is_sharing:
            if gat_attn_type == 'dot':
                gat = GATLayer(in_features, out_features, dropout, n_heads=n_heads, activation_type=activation_type)
            else:
                gat = GeneralGATLayer(in_features, out_features, dropout, n_heads=n_heads,
                                      alpha=0.2, activation_type='relu', residual=residual)
            for i in range(n_layers):
                self.gats.add_module("GAT-{}".format(i), gat)
        else:
            assert in_features == out_features
            for i in range(n_layers):
                if gat_attn_type == 'dot':
                    self.gats.add_module("GAT-{}".format(i), GATLayer(in_features, out_features,
                                                                      dropout, n_heads=n_heads,
                                                                      activation_type=activation_type))
                else:
                    self.gats.add_module("GAT-{}".format(i), GeneralGATLayer(in_features, out_features, dropout,
                                                                             n_heads=n_heads,
                                                                             alpha=leaky_relu_alpha,
                                                                             activation_type=activation_type,
                                                                             residual=residual))

        self.is_self = is_self
        self.residual = residual

    def forward(self, x):
        """

        :param x: [bsz, n, hsz]
        :return:
        """
        n = x.size(1)
        adj = self.adj.view(1, n, n)
        for layer in self.gats:
            out = layer(x, adj)
            if self.residual:
                out = out + x
            x = out

        return x.transpose(0, 1)


class GateGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, n_heads=1, gat_attn_type='dot', activate='none'):
        super(GateGATLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.key_projection = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.query_projection = nn.Linear(in_features=in_features, out_features=out_features, bias=False)

        if gat_attn_type == 'dot':
            self.value_projection = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
            self.d_k = out_features // n_heads
            # self.sqrt_d_k = math.sqrt(self.d_k) if n_heads > 1 else 1
            self.sqrt_d_k = math.sqrt(self.d_k)
            self.n_heads = n_heads
        elif gat_attn_type == 'general':
            self.linear_in = nn.Linear(in_features=out_features, out_features=out_features, bias=False)
            self.value_projection = nn.Linear(in_features=in_features, out_features=in_features, bias=False)

        self.softmax = nn.Softmax(dim=2)

        self.dropout = nn.Dropout(p=dropout)

        self.gate_linear = nn.Linear(in_features=2 * in_features, out_features=in_features)
        self.sigmoid = nn.Sigmoid()
        self.gat_attn_type = gat_attn_type

        if activate == 'relu':
            self.activation = nn.ReLU()
        elif activate == 'elu':
            self.activation = nn.ELU(alpha=0.1)
        elif activate == 'tanh':
            self.activation = nn.Tanh()
        elif activate == 'gelu':
            self.activation = GELU()
        self.activate = activate

    def dot_score(self, x, adj):
        bsz, n, hsz = x.size()
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        def shape_projection(tensor):
            b, l, d = tensor.size()
            return tensor.view(b, l, self.n_heads, self.d_k) \
                .transpose(1, 2).contiguous() \
                .view(b * self.n_heads, l, self.d_k)

        query_up = shape_projection(query)
        key_up = shape_projection(key)
        value_up = shape_projection(value)
        scaled = torch.bmm(query_up, key_up.transpose(1, 2))
        scaled = scaled / self.sqrt_d_k
        scaled = scaled.masked_fill_(~adj.bool(), -float('inf'))  # [bsz*n_heads, n, n]
        # print(scaled[0])
        attention = self.softmax(scaled)

        attention = self.dropout(attention)

        # [bsz, n_heads, n, n] [bsz, n_heads, n, d_k] -> [bsz, n_heads, n, d_k]
        c = torch.bmm(attention.view(-1, n, n), value_up)
        c = c.view(bsz, self.n_heads, n, self.d_k). \
            transpose(1, 2).contiguous().view(bsz, n, self.n_heads * self.d_k)

        if self.activate in ['relu', 'elu', 'tanh', 'gelu']:
            c = self.activation(c)

        return c

    def general_score(self, x, adj):
        query = self.query_projection(x)
        key = self.key_projection(x)

        query = self.linear_in(query)

        # [bsz, n_player+n_team, hsz] x [bsz, n_player+n_team, hsz] -> [bsz, n_player+n_team, n_player+n_team]
        scores = torch.bmm(query, key.transpose(1, 2))

        attention = scores.masked_fill_(~adj.bool(), -float('inf'))

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        c = torch.bmm(attention, self.value_projection(x))

        if self.activate in ['relu', 'elu', 'tanh', 'gelu']:
            c = self.activation(c)

        return c

    def forward(self, x, adj):
        """
        :param x: [bsz, n_player+n_team, hsz]
        :param adj: [bsz, n_player+n_team, n_player+n_team]
        :return:
        """
        if self.gat_attn_type == 'dot':
            c = self.dot_score(x, adj)
        elif self.gat_attn_type == 'general':
            c = self.general_score(x, adj)
        else:
            assert False

        gate = self.sigmoid(self.gate_linear(torch.cat([c, x], dim=2)))

        outputs = gate.mul(x) + (1 - gate).mul(c)
        return outputs


class GateGAT(nn.Module):
    def __init__(self, in_features, out_features, dropout,
                 n_layers=2, is_sharing=False, is_self=False, n_heads=1, player_player=True, gat_attn_type='dot',
                 using_global_node=False, activate='none'):
        super(GateGAT, self).__init__()

        assert n_layers > 0
        adj = get_adj_matrix(is_normed=False, is_self=is_self, player_player=player_player,
                             using_global_node=using_global_node)
        self.adj = nn.Parameter(adj)
        self.adj.requires_grad = False
        self.gats = nn.Sequential()

        if is_sharing:
            gat = GateGATLayer(in_features, out_features, dropout, n_heads=n_heads, gat_attn_type=gat_attn_type,
                               activate=activate)
            for i in range(n_layers):
                self.gats.add_module("GateGAT-{}".format(i), gat)
        else:
            if gat_attn_type == 'dot':
                assert in_features == out_features
            for i in range(n_layers):
                self.gats.add_module("GateGAT-{}".format(i), GateGATLayer(in_features, out_features,
                                                                          dropout, n_heads=n_heads,
                                                                          gat_attn_type=gat_attn_type,
                                                                          activate=activate))

        self.is_self = is_self

    def forward(self, x):
        """

        :param x: [bsz, n, hsz]
        :return:
        """
        n = x.size(1)
        adj = self.adj.view(1, n, n)
        for layer in self.gats:
            out = layer(x, adj)
            x = out

        return x.transpose(0, 1)


class HGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3, n_heads=1, gnn_activation_type='relu'):
        super(HGATLayer, self).__init__()

        self.player_query = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.team_query = nn.Linear(in_features=in_features, out_features=out_features, bias=False)

        self.player_key = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.team_key = nn.Linear(in_features=in_features, out_features=out_features, bias=False)

        self.player_val = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.team_val = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.player_out = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.team_out = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.activation = nn.ReLU() if gnn_activation_type == 'relu' else nn.Tanh()

        d_k = out_features // n_heads
        # num_types, num_relations, num_types
        self.relation_pri = nn.Parameter(torch.ones(2, 4, 2, n_heads))
        self.relation_pri.requires_grad = True
        self.relation_att = nn.Parameter(torch.Tensor(4, n_heads, d_k, d_k))
        # self.relation_att = nn.ModuleList(
        #     [nn.Linear(in_features=out_features, out_features=out_features, bias=False) for i in range(4)])
        self.relation_msg = nn.Parameter(torch.Tensor(4, n_heads, d_k, d_k))

        self.d_k = d_k
        self.sqrt_dk = math.sqrt(d_k)
        self.n_heads = n_heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, players, teams, player_adj, team_adj):
        """

        :param players: bsz, n_player, hsz
        :param teams: bsz, n_teams, hsz
        :param player_adj: bsz, n_player, n_player+n_team
        :param team_adj: bsz, n_teams, n_player+n_team
        :return:
        """
        bsz, n_player, hsz = players.size()
        _, n_team, _ = teams.size()

        query_players = self.player_query(players).view(bsz, n_player, self.n_heads, self.d_k)\
            .transpose(1, 2).contiguous().transpose(2, 3) # bsz, n_heads, d_k, n_player
        query_teams = self.team_query(teams).view(bsz, n_team, self.n_heads, self.d_k)\
            .transpose(1, 2).contiguous().transpose(2, 3)

        key_players = self.player_key(players).view(-1, self.n_heads, self.d_k)
        key_teams = self.team_key(teams).view(-1, self.n_heads, self.d_k)

        val_players = self.player_val(players).view(-1, self.n_heads, self.d_k)
        val_teams = self.team_val(teams).view(-1, self.n_heads, self.d_k)

        # player - player
        # calculate Attn_head = K W_rel_att Q
        # pla_pla_mat = torch.bmm(key_players.transpose(1, 0), self.relation_att[0]).transpose(1, 0)
        # K W_rel_att: [bsz*n_player, n_head, d_k] x [n_head, d_k, d_k] = [bsz*n_player, n_head, d_k]

        pla_pla_mat = torch.bmm(key_players.transpose(1, 0), self.relation_att[0]).transpose(1, 0).contiguous()
        pla_pla_mat = pla_pla_mat.view(bsz, n_player, self.n_heads, self.d_k).transpose(1, 2)

        pla_pla_attn = torch.matmul(pla_pla_mat, query_players).transpose(2, 3)  # bsz, n_heads, n_player, n_player
        pla_pla_attn = pla_pla_attn * self.relation_pri[0][0][0].expand(bsz, self.n_heads)\
            .view(bsz, self.n_heads, 1, 1) / self.sqrt_dk

        # [bsz*n_player, n_head, d_k] x [n_head, d_k, d_k] = [bsz*n_player, n_head, d_k]
        p_p_res_msg = torch.bmm(val_players.transpose(1, 0), self.relation_msg[0]).transpose(1, 0)

        # player - team
        pla_team_mat = torch.bmm(key_teams.transpose(1, 0), self.relation_att[1]).transpose(1, 0).contiguous()
        pla_team_mat = pla_team_mat.view(bsz, n_team, self.n_heads, self.d_k).transpose(1, 2)

        pla_team_attn = torch.matmul(pla_team_mat, query_players).transpose(2, 3)
        pla_team_attn = pla_team_attn * self.relation_pri[0][1][1].expand(bsz, self.n_heads)\
            .reshape(bsz, self.n_heads, 1, 1) / self.sqrt_dk

        p_t_res_msg = torch.bmm(val_teams.transpose(1, 0), self.relation_msg[1]).transpose(1, 0)
        # team - player
        team_pla_mat = torch.bmm(key_players.transpose(1, 0), self.relation_att[2]).transpose(1, 0).contiguous()
        team_pla_mat = team_pla_mat.view(bsz, n_player, self.n_heads, self.d_k).transpose(1, 2)

        team_pla_attn = torch.matmul(team_pla_mat, query_teams).transpose(2, 3)
        team_pla_attn = team_pla_attn * self.relation_pri[1][2][0].expand(bsz, self.n_heads)\
            .reshape(bsz, self.n_heads, 1, 1) / self.sqrt_dk

        t_p_res_msg = torch.bmm(val_players.transpose(1, 0), self.relation_msg[2]).transpose(1, 0)
        # team team
        team_team_mat = torch.bmm(key_teams.transpose(1, 0), self.relation_att[3]).transpose(1, 0).contiguous()
        team_team_mat = team_team_mat.view(bsz, n_team, self.n_heads, self.d_k).transpose(1, 2)

        team_team_attn = torch.matmul(team_team_mat, query_teams).transpose(2, 3)
        team_team_attn = team_team_attn * self.relation_pri[1][3][1].expand(bsz, self.n_heads)\
            .reshape(bsz, self.n_heads, 1, 1) / self.sqrt_dk

        t_t_res_msg = torch.bmm(val_teams.transpose(1, 0), self.relation_msg[3]).transpose(1, 0)

        # concat
        # the query is player
        player_attn = torch.cat([pla_pla_attn, pla_team_attn], dim=3)
        player_attn = player_attn.masked_fill_(~player_adj.bool(), -float('inf'))
        player_score = self.softmax(player_attn)
        player_score = self.dropout(player_score)  # bsz, n_heas, n_player, -1

        player_val = torch.cat([p_p_res_msg.view(bsz, n_player, self.n_heads, -1),
                                p_t_res_msg.view(bsz, n_team, self.n_heads, -1)],
                               dim=1).transpose(1, 2).contiguous()
        player_rep = torch.matmul(player_score, player_val)

        # the query is team
        team_attn = torch.cat([team_pla_attn, team_team_attn], dim=3)

        team_attn = team_attn.masked_fill_(~team_adj.bool(), -float('inf'))
        team_score = self.softmax(team_attn)
        team_score = self.dropout(team_score)

        team_val = torch.cat([t_p_res_msg.view(bsz, n_player, self.n_heads, -1),
                              t_t_res_msg.view(bsz, n_team, self.n_heads, -1)],
                             dim=1).transpose(1, 2).contiguous()

        team_rep = torch.matmul(team_score, team_val)  # bsz, n_head x n_team x d_k

        player_rep = player_rep.transpose(1, 2).reshape(bsz, n_player, -1)
        team_rep = team_rep.transpose(1, 2).reshape(bsz, n_team, -1)

        player_rep = self.player_out(self.activation(player_rep))
        team_rep = self.team_out(self.activation(team_rep))

        return player_rep, team_rep
    

class HGAT(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3, n_heads=1, n_layers=1, is_self=False,
                 is_sharing=False, gnn_activation_type='relu', player_player=True):
        super(HGAT, self).__init__()
        assert n_layers > 0
        adj = get_adj_matrix(is_normed=False, is_self=is_self, player_player=player_player) # n x n
        self.adj = nn.Parameter(adj)
        self.adj.requires_grad = False
        self.hgats = nn.Sequential()
        if is_sharing:
            hgat = HGATLayer(in_features=in_features, out_features=out_features, dropout=dropout, n_heads=n_heads,
                             gnn_activation_type=gnn_activation_type)
            for i in range(n_layers):
                self.hgats.add_module("HGAT-{}".format(i), hgat)
        else:
            assert in_features == out_features
            for i in range(n_layers):
                self.hgats.add_module("HGAT-{}".format(i), HGATLayer(in_features=in_features, out_features=out_features,
                                                                     dropout=dropout, n_heads=n_heads,
                                                                     gnn_activation_type=gnn_activation_type))

        self.is_self = is_self
        self.n_heads = n_heads

    def forward(self, x):
        """

        :param x: [bsz, n, hsz]
        :return:
        """
        bsz, n, hsz = x.size()
        assert n == 28
        players = x.narrow(dim=1, start=0, length=26)
        teams = x.narrow(dim=1, start=26, length=2)

        player_adj = self.adj.narrow(dim=0, start=0, length=26).view(1, 1, players.size(1), n)
        team_adj = self.adj.narrow(dim=0, start=26, length=2).view(1, 1, teams.size(1), n)

        for layer in self.hgats:
            out_players, out_teams = layer(players, teams, player_adj, team_adj)

            players = players + players
            teams = teams + out_teams

        outputs = torch.cat([players, teams], dim=1)
        return outputs.transpose(0, 1)








