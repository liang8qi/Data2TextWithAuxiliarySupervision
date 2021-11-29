import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(-1) == 1:
            return z
        mu = torch.mean(z, dim=-1, keepdim=True)
        sigma = torch.std(z, dim=-1, keepdim=True)
        ln_out = (z - mu) / (sigma + self.eps)
        ln_out = ln_out.mul(self.a_2) + self.b_2
        return ln_out


class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    Inputs are a 3d Variable whose last dimension is the same length
    as the list.
    Outputs are the result of applying modules to inputs elementwise.
    An optional merge parameter allows the outputs to be reduced to a
    single Variable.
    """

    def __init__(self, merge=None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp'], merge
        self.merge = merge

        super(Elementwise, self).__init__(*args)

    def forward(self, input):
        # assert False
        inputs = [feat.squeeze(2) for feat in input.split(1, dim=2)]
        assert len(self) == len(inputs)
        outputs = [f(x) for f, x in zip(self, inputs)]

        if self.merge == 'first':
            return outputs[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(outputs, 2)
        elif self.merge == 'sum':
            return sum(outputs)
        else:
            return outputs


class CharHighwayNet(nn.Module):
    def __init__(self, dim, activation='tanh'):
        super(CharHighwayNet, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(dim))
        self.token = nn.Linear(dim, dim, bias=False)
        self.char = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, rep_word, rep_char):
        n, bsz, hsz = rep_word.size()
        w = self.W.view(1, 1, 1, hsz).repeat(606, bsz, 1, 1)

        cated = torch.cat([rep_word.view(n, bsz, 1, hsz), rep_char.view(n, bsz, 1, hsz)], dim=2)
        f_rep_word = self.token(rep_word.view(n, bsz, 1, hsz))
        f_rep_char = self.char(rep_char.view(n, bsz, 1, hsz))
        feat = torch.cat([f_rep_word, f_rep_char], dim=2)

        scores = torch.matmul(feat, w.transpose(3, 2))  # 606, bsz, 2, 1

        attn = self.softmax(scores)

        outputs = torch.matmul(cated.transpose(3, 2), attn).view(n, bsz, hsz)

        # z = t*(1-t)*rep_word + (1-t)*rep_char
        # z = (1-t)*rep_word + t*rep_char
        return outputs


"""
class CharHighwayNet(nn.Module):
    def __init__(self, dim, activation='tanh'):
        super(CharHighwayNet, self).__init__()
        self.W_T = nn.Linear(in_features=2*dim, out_features=2*dim, bias=True)
        self.W_H = nn.Linear(in_features=2*dim, out_features=2*dim, bias=True)
        self.sigmoid = nn.Sigmoid()

        if activation == 'tanh':
            self.g = nn.Tanh()
        elif activation == 'relu':
            self.g = nn.ReLU()
        else:
            assert False

    def forward(self, rep_word, rep_char):
        h = torch.cat([rep_word, rep_char], dim=-1)

        t = self.sigmoid(self.W_T(h))
        z = t*self.g(self.W_H(h)) + (1-t)*h
        # z = t*(1-t)*rep_word + (1-t)*rep_char
        # z = (1-t)*rep_word + t*rep_char
        return z
"""
"""
class CharHighwayNet(nn.Module):
    def __init__(self, dim, activation='tanh'):
        super(CharHighwayNet, self).__init__()
        self.W_T = nn.Linear(in_features=dim, out_features=dim, bias=True)
        # self.W_H = nn.Linear(in_features=2*dim, out_features=2*dim, bias=True)
        self.sigmoid = nn.Sigmoid()
  
        if activation == 'tanh':
            self.g = nn.Tanh()
        elif activation == 'relu':
            self.g = nn.ReLU()
        else:
            assert False


    def forward(self, rep_word, rep_char):
        # h = torch.cat([rep_word, rep_char], dim=-1)

        t = self.sigmoid(self.W_T(rep_word))
        # z = t*self.g(self.W_H(h)) + (1-t)*h
        # z = t*(1-t)*rep_word + (1-t)*rep_char
        z = (1-t)*rep_word + t*rep_char
        return z

"""