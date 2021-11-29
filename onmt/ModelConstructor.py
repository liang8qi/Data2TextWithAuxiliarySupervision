"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.Models import NMTModel, RNNEncoder, \
                        HierarchicalEncoder, \
                        StdRNNDecoder, InputFeedRNNDecoder
from onmt.modules import Embeddings, ImageEncoder, CopyGenerator, \
                         TransformerDecoder, \
                         CNNEncoder, CNNDecoder, AudioEncoder
from onmt.Utils import use_gpu

from auxiliary.ptr_decoder import PtrDecoder
from auxiliary.table_encoder import TableEncoder


def make_embeddings(opt, word_dict, feature_dicts, for_encoder=True, word_same_with_feat=False,
                    ):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    feat_vec_size = opt.feat_vec_size
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[onmt.io.PAD_WORD]
    num_word_embeddings = len(word_dict)
    feats_padding_idx = [feat_dict.stoi[onmt.io.PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]

    if type(opt.ngram_filter_sizes[0]) == str:
        opt.ngram_filter_sizes = tuple([int(i) for i in opt.ngram_filter_sizes])

    main_emb = Embeddings(word_vec_size=embedding_dim,
                          position_encoding=opt.position_encoding,
                          feat_merge=opt.feat_merge,
                          feat_vec_exponent=opt.feat_vec_exponent,
                          feat_vec_size=feat_vec_size,
                          dropout=opt.dropout,
                          word_same_with_feat=word_same_with_feat,
                          word_padding_idx=word_padding_idx,
                          feat_padding_idx=feats_padding_idx,
                          word_vocab_size=num_word_embeddings,
                          feat_vocab_sizes=num_feat_embeddings,
                    )



    return main_emb


def make_encoder(opt, embeddings, stage1=True, basic_enc_dec=False, use_hier_hist=False):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
        stage1: stage1 encoder
    """

    assert basic_enc_dec
    char_embedding=None

    if opt.encoder_type == 'my':

        return TableEncoder(meta=opt.hier_meta, num_layers=opt.enc_layers1, embeddings=embeddings,
                            emb_size=opt.src_word_vec_size, attn_hidden=opt.attn_hidden,
                            bidirectional=opt.hier_bidirectional, hidden_size=opt.hier_rnn_size, dropout=opt.dropout,
                            attn_type=opt.global_attention, table_attn_type=opt.table_attn_type,
                            two_dim_score=opt.two_dim_score, no_fusion=opt.no_fusion,
                            n_heads=opt.table_att_n_heads, is_fnn=opt.is_fnn, is_layer_norm=opt.is_layer_norm,
                            row_aggregation=opt.row_aggregation, aggr_att_type=opt.aggr_att_type,
                            row_rep_type=opt.row_rep_type, gnn_type=opt.gnn_type, gnn_share=opt.gnn_share,
                            gnn_n_layers=opt.gnn_n_layers,
                            gat_dropout=opt.gat_dropout, gat_is_self=opt.gat_is_self,
                            gat_n_heads=opt.gat_n_heads, gnn_activation_type=opt.gnn_activation_type,
                            player_player=opt.player_player, row_agg_is_param=opt.row_agg_is_param,
                            gat_attn_type=opt.gat_attn_type,
                            residual=opt.gat_residual, leaky_relu_alpha=opt.gat_leaky_relu_alpha,
                            using_global_node=opt.using_global_node)
    else:
        return HierarchicalEncoder(opt.hier_meta, opt.enc_layers1, embeddings,
                                   opt.src_word_vec_size, opt.attn_hidden,
                                   opt.hier_rnn_type, opt.hier_bidirectional,
                                   opt.hier_rnn_size, dropout=opt.dropout,
                                   attn_type=opt.global_attention, two_dim_score=opt.two_dim_score,
                                   hier_history_seq_type=opt.hier_history_seq_type,
                                   hier_history_seq_window=opt.hier_history_seq_window,
                                   hier_num_layers=opt.hier_num_layers, hier_hist_attn_type=opt.hier_hist_attn_type,
                                   hier_hist_attn_pos_type=opt.hier_hist_attn_pos_type, use_hier_hist=use_hier_hist)


def make_decoder(opt, embeddings, stage1, basic_enc_dec):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
        stage1: stage1 decoder
    """
    if opt.decoder_type2 == 'transformer':
        return TransformerDecoder(num_layers=opt.dec_layers2, hidden_size=opt.rnn_size,
                                  attn_type=opt.global_attention, copy_attn=opt.copy_attn,
                                  dropout=opt.dropout, embeddings=embeddings)
    return InputFeedRNNDecoder(opt.rnn_type, opt.brnn2,
                                   opt.dec_layers2, opt.rnn_size,
                                   opt.global_attention,
                                   opt.coverage_attn,
                                   opt.context_gate,
                                   True,
                                   opt.dropout,
                                   embeddings,
                                   opt.reuse_copy_attn,
                                   hier_attn=True)


def load_test_model(opt, dummy_opt, model_src, stage1=False):
    # opt_model = opt.model2
    opt_model = model_src
    checkpoint = torch.load(opt_model,
                            map_location=lambda storage, loc: storage)

    model_opt = checkpoint['opt']

    fields = onmt.io.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type, cal_indices=model_opt.cal_indices)

    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    model = make_base_model(model_opt, fields,
                            use_gpu(opt), checkpoint, False, True)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def make_base_model(model_opt, fields, gpu, checkpoint=None, stage1=True, basic_enc_dec=False):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """

    src = "src"
    tgt = "tgt"
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make encoder.
    if model_opt.model_type == "text":
        src_dict = fields[src].vocab

        feature_dicts = onmt.io.collect_feature_vocabs(fields, src)
        src_embeddings = make_embeddings(model_opt, src_dict,
                                         feature_dicts,
                                         word_same_with_feat=model_opt.word_same_with_feat)
        encoder = make_encoder(model_opt, src_embeddings, stage1, basic_enc_dec,
                               use_hier_hist=model_opt.use_hier_hist)
    elif model_opt.model_type == "img":
        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    # Make decoder.
    tgt_dict = fields[tgt].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, tgt)
    tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.make_embedding_first[0].emb_luts[0].weight = src_embeddings.make_embedding_first[0].emb_luts[0].weight
        # tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    decoder = make_decoder(model_opt, tgt_embeddings, stage1 and not basic_enc_dec, basic_enc_dec)

    num_ranking_decoder = None
    imp_ranking_decoder = None
    if model_opt.enable_number_ranking:

        num_ranking_decoder = PtrDecoder(emb_dim=model_opt.src_word_vec_size, hsz=model_opt.src_word_vec_size,
                                     rnn_type=model_opt.sort_rnn_type, is_input_feed=model_opt.sort_is_input_feed,
                                     dropout=model_opt.sort_dropout, att_type=model_opt.sort_att_type,
                                     sort_type='col', is_mask=model_opt.sort_is_mask)

    if model_opt.enable_importance_ranking:
        imp_ranking_decoder = PtrDecoder(emb_dim=model_opt.src_word_vec_size, hsz=model_opt.src_word_vec_size,
                                     rnn_type=model_opt.sort_rnn_type, is_input_feed=model_opt.sort_is_input_feed,
                                     dropout=model_opt.sort_dropout, att_type=model_opt.sort_att_type,
                                     sort_type='row', is_mask=model_opt.sort_is_mask)

    # Make NMTModel(= encoder + decoder).
    model = NMTModel(encoder, decoder, num_ranking_decoder=num_ranking_decoder, imp_ranking_decoder=imp_ranking_decoder)
    model.model_type = model_opt.model_type

    # Make Generator.
    generator = CopyGenerator(model_opt.rnn_size,
                                  fields["tgt"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        try:
            model.load_state_dict(checkpoint['model'])
        except RuntimeError as err:
            print("There are some errors during loading pre-trained model:")
            print(err)
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                if not p.requires_grad:
                    continue
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model
