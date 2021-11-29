import torch
import torch.nn as nn
import math

from onmt.Utils import aeq, sequence_mask


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, dim, dropout):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.k_lin = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.v_lin = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.out_lin = nn.Linear(in_features=dim, out_features=dim, bias=True)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, kv=None, memory_lengths=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        # one step input
        if query.dim() == 2:
            one_step = True
            query = query.unsqueeze(1)
        else:
            one_step = False

        bs, qlen, dim = query.size()
        if kv is None:
            klen = qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = dim // n_heads

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2).contiguous()

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head).transpose(0, 1).contiguous()

        q = shape(self.q_lin(query))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(query))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(query))  # (bs, n_heads, qlen, dim_per_head)
        else:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, qlen, klen)
        # mask = (mask == 0).view(mask_reshape).expand_as(scores)               # (bs, n_heads, qlen, klen)
        # scores.masked_fill_(mask, -float('inf'))                              # (bs, n_heads, qlen, klen)

        assert memory_lengths is not None

        mask = sequence_mask(memory_lengths)
        mask = mask.unsqueeze(1)  # Make it broadcastable.
        # mask the time step of self
        mask = mask.repeat(1, klen, 1)
        mask_self_index = list(range(klen))
        mask[:, mask_self_index, mask_self_index] = 0

        mask = mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        scores = scores.masked_fill_(~mask, value=-float('inf'))

        weights = self.softmax(scores.float()).type_as(scores)  # (bs, n_heads, qlen, klen)
        weights = self.dropout(weights)  # (bs, n_heads, qlen, klen)
        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)
        outputs = self.out_lin(context)

        if one_step:
            outputs = outputs.unsqueeze(0)
            weights = weights.unsqueeze(0)
            context = context.unsqueeze(0)

        return outputs, weights, context


class TransformerFFN(nn.Module):

    def __init__(self, in_dim, dim_hidden, out_dim, dropout, gelu_activation=True):
        super().__init__()
        self.lin1 = nn.Linear(in_features=in_dim, out_features=dim_hidden)
        self.lin2 = nn.Linear(in_features=dim_hidden, out_features=out_dim)
        self.act = gelu if gelu_activation else nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        x = self.lin1(features)
        x = self.act(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
