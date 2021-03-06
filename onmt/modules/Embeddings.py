import torch
import torch.nn as nn
from torch.autograd import Variable

from onmt.modules import Elementwise
from onmt.Utils import aeq
from onmt.Models import RNNEncoder


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.arange(0, max_len).unsqueeze(1).expand(max_len, dim)
        div_term = 1 / torch.pow(10000, torch.arange(0, dim * 2, 2) / dim)
        pe = pe * div_term.expand_as(pe)
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb):
        # We must wrap the self.pe in Variable to compute, not the other
        # way - unwrap emb(i.e. emb.datasets). Otherwise the computation
        # wouldn't be watched to build the compute graph.
        emb = emb + Variable(self.pe[:emb.size(0), :1, :emb.size(2)]
                             .expand_as(emb), requires_grad=False)
        emb = self.dropout(emb)
        return emb

class dumpEmb(object):
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size


class ProcessChar(nn.Module):
    """
        args:
    """
    def __init__(self, emb_luts):
        super(ProcessChar, self).__init__()
        self.emb_luts = emb_luts

    def forward(self, src):
        # if isinstance(src, tuple):
        #     src = src[0]
        return self.emb_luts(src)



class Embeddings(nn.Module):
    """
    Words embeddings for encoder/decoder.

    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.


    .. mermaid::

       graph LR
          A[Input]
          C[Feature 1 Lookup]
          A-->B[Word Lookup]
          A-->C
          A-->D[Feature N Lookup]
          B-->E[MLP/Concat]
          C-->E
          D-->E
          E-->F[Output]

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        feats_padding_idx (list of int): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes ([int], optional): list of size of dictionary
                                    of embeddings for each feature.

        position_encoding (bool): see :obj:`onmt.modules.PositionalEncoding`

        feat_merge (string): merge action for the features embeddings:
                    concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
                    embedding size is N^feat_dim_exponent, where N is the
                    number of values of feature takes.
        feat_vec_size (int): embedding dimension for features when using
                    `-feat_merge mlp`
        dropout (float): dropout probability.
    """
    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
                 position_encoding=False,
                 feat_merge="concat",
                 feat_vec_exponent=0.7, feat_vec_size=-1,
                 feat_padding_idx=[],
                 feat_vocab_sizes=[],
                 dropout=0,
                 word_same_with_feat=False,
                 emb_for_hier_hist=False, external_embedding=None):

        self.word_padding_idx = word_padding_idx

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]

        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        if word_same_with_feat and feat_vec_size > 0:
            emb_dims[0] = feat_vec_size
        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        if external_embedding is not None:
            assert emb_for_hier_hist
            embeddings = external_embedding
        else:
            emb_params = zip(vocab_sizes, emb_dims, pad_indices)
            embeddings = [nn.Embedding(vocab, dim, padding_idx=pad)
                          for vocab, dim, pad in emb_params]

        self.store_emb_info = embeddings
        # for name, para in embeddings[0].named_parameters():
        #     print("{}: {}".format(name, torch.sum(para)))
        emb_luts = Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else word_vec_size)

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(Embeddings, self).__init__()
        self.make_embedding_first = nn.Sequential()
        self.make_embedding_second = nn.Sequential()
        if emb_for_hier_hist:
            in_dim = sum(emb_dims)
            out_dim = word_vec_size
            mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            self.make_embedding_first.add_module('emb_luts', emb_luts)
            self.make_embedding_first.add_module('mlp', mlp)
        else:
            # self.make_embedding.add_module('emb_luts', emb_luts)
            char_proc = ProcessChar(emb_luts)
            self.make_embedding_first.add_module('char_proc', char_proc)

            if feat_merge == 'mlp' and len(feat_vocab_sizes) > 0:
                in_dim = sum(emb_dims)
                out_dim = word_vec_size

                mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
                # for name, para in mlp.named_parameters():
                #     print("{}: {}".format(name, torch.sum(para)))
                self.make_embedding_second.add_module('mlp', mlp)

        if position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding_second.add_module('pe', pe)

    @property
    def word_lut(self):
        if isinstance(self.make_embedding[0], Elementwise):
            return self.make_embedding[0][0]
        else:
            return self.make_embedding[0].emb_luts[0]

    @property
    def emb_luts(self):
        if isinstance(self.make_embedding_first[0], Elementwise):
            return self.make_embedding_first[0]
        else:
            return self.make_embedding_first[0].emb_luts

    def load_pretrained_vectors(self, emb_file, fixed):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
          fixed (bool) : if true, embeddings are not updated
        """
        if emb_file:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)
            if fixed:
                self.word_lut.weight.requires_grad = False

    @property
    def get_feat_emb(self):
        return self.store_emb_info[1:]

    def forward(self, input):
        """
        Computes the embeddings for words and features.

        Args:
            
        Return:
            `FloatTensor`: word embeddings `[len x batch x embedding_size]`
        """
        if isinstance(input, tuple):
            check_input = input[0]
            assert False
        else:
            check_input = input
        in_length, in_batch, nfeat = check_input.size()

        aeq(nfeat, len(self.emb_luts))

            # assert False
        emb = self.make_embedding_first(input)

        emb = self.make_embedding_second(emb)

        out_length, out_batch, emb_size = emb.size()
        aeq(in_length, out_length)
        aeq(in_batch, out_batch)
        aeq(emb_size, self.embedding_size)

        return emb


def get_masks(slen, lengths):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    # assert lengths.max().item() <= slen
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    # mask = alen >= lengths[:, None]
    mask = alen < lengths[:, None]
    return mask


