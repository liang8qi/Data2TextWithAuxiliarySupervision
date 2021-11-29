import argparse
from onmt.modules.SRU import CheckSRU

import ast


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    # Embedding Options
    group = parser.add_argument_group('Model-Embeddings')
    group = parser.add_argument_group('Model-Embeddings')
    group.add_argument('-src_word_vec_size', type=int, default=500,
                       help='Word embedding size for src.')
    group.add_argument('-tgt_word_vec_size', type=int, default=500,
                       help='Word embedding size for tgt.')
    group.add_argument('-word_vec_size', type=int, default=600,
                       help='Word embedding size for src and tgt.')
    group.add_argument('-attn_hidden', type=int, default=64,
                       help='Attn hidden size for self attention on input')

    # encoder type
    group.add_argument('-encoder_type', type=str, default='my', choices=['my', 'hier'])
    # hier model 1 options
    group.add_argument("-no_fusion", type=ast.literal_eval, default=False)
    group.add_argument('-two_dim_score', type=str, default='mlp', choices=["general", "mlp", "dot", None],
                       help='two_dim_score')
    # emb_gate
    group.add_argument('-num_filters', type=int, default=128)
    group.add_argument('-ngram_filter_sizes', type=tuple, default=(2, 3, 4, 5))
    group.add_argument('-word_same_with_feat', type=ast.literal_eval, default=False)
    # ------ GNN
    # group.add_argument('-using_gnn', type=ast.literal_eval, default=True)
    group.add_argument('-row_rep_type', type=str, default='gnn', choices=['gnn', 'attn', 'oral'])
    group.add_argument('-gnn_type', type=str, default="gategat", choices=['gcn', 'gat', 'gategat', 'hgat'])
    group.add_argument('-gnn_share', type=ast.literal_eval, default=False)
    group.add_argument('-gnn_n_layers', type=int, default=2)
    group.add_argument('-gnn_activation_type', type=str, default='none', choices=['relu', 'tanh', 'elu', 'gelu', 'none'])
    group.add_argument('-gat_n_heads', type=int, default=4)
    group.add_argument('-player_player', type=ast.literal_eval, default=True)
    group.add_argument('-gat_attn_type', type=str, default='dot', choices=['dot', 'general'],
                       help='using only for GateGAT')
    group.add_argument('-gat_residual', type=ast.literal_eval, default=True)
    group.add_argument('-gat_leaky_relu_alpha', type=float, default=0.2)
    group.add_argument("-gat_dropout", type=float, default=0.3)
    group.add_argument('-gat_is_self', type=ast.literal_eval, default=False)
    group.add_argument('-using_global_node', type=ast.literal_eval, default=False)
    # ------
    # hier history
    group.add_argument('-use_hier_hist', type=ast.literal_eval, default=False)
    # hier seq history
    group.add_argument('-hier_history_seq_type', type=str, default='SA',
                       choices=['rnn', 'RNN', "SA", "sa", None])
    group.add_argument('-hier_history_seq_window', type=int, default=3,
                       help="""hier_history_seq_window""")
    group.add_argument('-hier_num_layers', type=int, default=None)
    group.add_argument('-hier_hist_attn_type', type=str, default='mlp',
                       choices=['dot', 'general', 'mlp', None])
    group.add_argument('-hier_hist_attn_pos_type', type=str, default='posEmb',
                       choices=["posEmb", "posEncoding", None])

    # hier_meta
    group.add_argument('-hier_meta', type=str, default='rotowire/hier_meta.json',
                       help="""hier meta path""")
    group.add_argument('-hier_rnn_type', type=str, default='LSTM',
                       choices=['LSTM', 'GRU'],
                       action=CheckSRU,
                       help="""The gate type to use in the hier RNNs""")
    group.add_argument('-hier_rnn_size', type=int, default=600,
                       help='Size of hier rnn hidden states')
    group.add_argument('-hier_bidirectional', default=True, type=ast.literal_eval,
                       help="""Activate biRNN for hier model""")

    #
    group.add_argument('-cal_indices', default=True, type=ast.literal_eval)

    # content selection gate 

    group.add_argument('-share_decoder_embeddings', action='store_true',
                       help="""Use a shared weight matrix for the input and
                       output word  embeddings in the decoder.""")
    group.add_argument('-share_embeddings', type=ast.literal_eval, default=False,
                       help="""Share the word embeddings between encoder
                       and decoder. Need to use shared dictionary for this
                       option.""")
    group.add_argument('-position_encoding', action='store_true',
                       help="""Use a sin to mark relative words positions.
                       Necessary for non-RNN style models.
                       """)

    group = parser.add_argument_group('Model-Embedding Features')
    group.add_argument('-feat_merge', type=str, default='mlp',
                       choices=['concat', 'sum', 'mlp'],
                       help="""Merge action for incorporating features embeddings.
                       Options [concat|sum|mlp].""")
    group.add_argument('-feat_vec_size', type=int, default=100,
                       help="""If specified, feature embedding sizes
                       will be set to this. Otherwise, feat_vec_exponent
                       will be used.""")
    group.add_argument('-feat_vec_exponent', type=float, default=0.7,
                       help="""If -feat_merge_size is not set, feature
                       embedding sizes will be set to N^feat_vec_exponent
                       where N is the number of values the feature takes.""")

    # Encoder-Deocder Options
    group = parser.add_argument_group('Model- Encoder-Decoder')
    group.add_argument('-row_aggregation', type=str, default='attention', choices=['mean', 'attention'])
    group.add_argument('-row_agg_is_param', type=ast.literal_eval, default=False)
    group.add_argument('-aggr_att_type', type=str, default='mlp', choices=['dot', 'mlp'])
    group.add_argument('-model_type', default='text',
                       help="""Type of source model to use. Allows
                       the system to incorporate non-text inputs.
                       Options are [text|img|audio].""")

    group.add_argument('-encoder_type1', type=str, default='mean',
                       choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn'],
                       help="""Type of encoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|brnn|mean|transformer|cnn].""")
    group.add_argument('-decoder_type1', type=str, default='rnn',
                       choices=['rnn', 'transformer', 'cnn', 'pointer'],
                       help="""Type of decoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|transformer|cnn].""")
    group.add_argument('-encoder_type2', type=str, default='mean',
                       choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn'],
                       help="""Type of encoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|brnn|mean|transformer|cnn].""")
    group.add_argument('-decoder_type2', type=str, default='rnn',
                       choices=['rnn', 'transformer', 'cnn', 'pointer'],
                       help="""Type of decoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|transformer|cnn].""")

    group.add_argument('-layers', type=int, default=-1,
                       help='Number of layers in enc/dec.')
    group.add_argument('-enc_layers1', type=int, default=2,
                       help='Number of layers in the encoder')
    group.add_argument('-dec_layers1', type=int, default=2,
                       help='Number of layers in the decoder')
    group.add_argument('-enc_layers2', type=int, default=2,
                       help='Number of layers in the encoder')
    group.add_argument('-dec_layers2', type=int, default=2,
                       help='Number of layers in the decoder')
    group.add_argument('-rnn_size', type=int, default=600,
                       help='Size of rnn hidden states')
    group.add_argument('-cnn_kernel_width', type=int, default=3,
                       help="""Size of windows in the cnn, the kernel_size is
                       (cnn_kernel_width, 1) in conv layer""")

    group.add_argument('-input_feed', type=int, default=1,
                       help="""Feed the context vector at each time step as
                       additional input (via concatenation with the word
                       embeddings) to the decoder.""")
    group.add_argument('-bridge', action="store_true",
                       help="""Have an additional layer between the last encoder
                       state and the first decoder state""")
    group.add_argument('-rnn_type', type=str, default='LSTM',
                       choices=['LSTM', 'GRU', 'SRU'],
                       action=CheckSRU,
                       help="""The gate type to use in the RNNs""")
    # group.add_argument('-residual',   action="store_true",
    #                     help="Add residual connections between RNN layers.")

    group.add_argument('-brnn', action=DeprecateAction,
                       help="Deprecated, use `encoder_type`.")
    group.add_argument('-brnn_merge', default='concat',
                       choices=['concat', 'sum'],
                       help="Merge action for the bidir hidden states")

    group.add_argument('-context_gate', type=str, default=None,
                       choices=['source', 'target', 'both'],
                       help="""Type of context gate to use.
                       Do not select for no context gate.""")

    # Attention options
    group = parser.add_argument_group('Model- Attention')
    group.add_argument('-global_attention', type=str, default='general',
                       choices=['dot', 'general', 'mlp'],
                       help="""The attention type to use:
                       dotprod or general (Luong) or MLP (Bahdanau)""")
    group.add_argument('-table_attn_type', type=str, default='general',
                       choices=['dot', 'general', 'mlp', 'mul_head'],
                       help="""The attention type to use:
                           dotprod or general (Luong) or MLP (Bahdanau)""")
    group.add_argument('-table_att_n_heads', type=int, default=4)
    group.add_argument('-is_fnn', type=ast.literal_eval, default=False)
    group.add_argument('-is_layer_norm', type=ast.literal_eval, default=False)

    # Genenerator and loss options.
    group.add_argument('-alpha_nr', default=1.0, type=float, help='the ')
    group.add_argument('-alpha_ir', default=0.35, type=float)
    group.add_argument('-copy_attn', default=True, type=ast.literal_eval,
                       help='Train copy attention layer.')
    group.add_argument('-copy_attn_force', action="store_true",
                       help='When available, train to copy.')
    group.add_argument('-reuse_copy_attn', default=True, type=ast.literal_eval,
                       help="Reuse standard attention for copy")
    group.add_argument('-copy_loss_by_seqlength', default=False, type=ast.literal_eval,
                       help="Divide copy loss by length of sequence")
    group.add_argument('-coverage_attn', action="store_true",
                       help='Train a coverage attention layer.')
    group.add_argument('-lambda_coverage', type=float, default=1,
                       help='Lambda value for coverage.')

    group.add_argument('-stage1', action="store_true",
                       help="Stage1 pre process")

    # whether using sort
    group.add_argument('-enable_number_ranking', default=False, type=ast.literal_eval)
    group.add_argument('-enable_importance_ranking', default=False, type=ast.literal_eval)
    group.add_argument('-sort_rnn_type', default='lstm', type=str)
    group.add_argument('-sort_is_input_feed', default=False, type=ast.literal_eval)
    group.add_argument('-sort_dropout', default=0.3, type=float)
    group.add_argument('-sort_att_type', default='mlp', type=str)
    group.add_argument('-sort_is_mask', default=False, type=ast.literal_eval)


def preprocess_opts(parser):
    # Data options
    group = parser.add_argument_group('Data')
    group.add_argument('-data_type', default="text",
                       help="""Type of the source input.
                       Options are [text|img].""")
    group.add_argument('-cal_indices', default=True, type=ast.literal_eval)
    group.add_argument('-save_data', default='',
                       type=str,
                       help="Output file for the prepared datasets")

    group.add_argument('-process_hier_meta', type=str, default='',
                       help="""hier meta path""")
    group.add_argument('-train',
                       help="Path to the training datasets")
    group.add_argument('-valid',
                       help="Path to the validation datasets")

    group.add_argument('-train_src', default='', type=str,
                       help="Path to the training source datasets")

    group.add_argument('-train_src_hist', default='', type=str,
                       help="Path to the train_src1_hist")

    group.add_argument('-train_tgt', default='', type=str,
                       help="Path to the training target datasets")
    group.add_argument('-valid_src', default='', type=str,
                       help="Path to the validation source datasets")

    group.add_argument('-valid_src_hist', default='', type=str,
                       help="Path to the valid_src1_hist")

    group.add_argument('-valid_tgt', default='', type=str,
                       help="Path to the validation target datasets")
    group.add_argument('-train_ptr', default='', type=str,
                       help="Path to the training pointers datasets")

    group.add_argument('-src_dir', default="",
                       help="Source directory for image or audio files.")

    group.add_argument('-max_shard_size', type=int, default=0,
                       help="""For text corpus of large volume, it will
                       be divided into shards of this size to preprocess.
                       If 0, the datasets will be handled as a whole. The unit
                       is in bytes. Optimal value should be multiples of
                       64 bytes.""")

    group.add_argument('-players_per_team', type=int, default=13,
                       help="""Max players per team""")

    group.add_argument('-entity_frequent_src', type=str, default=None)
    # Dictionary options, for
    # text corpus

    group = parser.add_argument_group('Vocab')
    group.add_argument('-src_vocab',
                       help="Path to an existing source vocabulary")
    group.add_argument('-tgt_vocab',
                       help="Path to an existing target vocabulary")
    group.add_argument('-features_vocabs_prefix', type=str, default='',
                       help="Path prefix to existing features vocabularies")
    group.add_argument('-src_vocab_size', type=int, default=50000,
                       help="Size of the source vocabulary")
    group.add_argument('-tgt_vocab_size', type=int, default=50000,
                       help="Size of the target vocabulary")

    group.add_argument('-src_words_min_frequency', type=int, default=0)
    group.add_argument('-tgt_words_min_frequency', type=int, default=0)

    group.add_argument('-dynamic_dict', default=True, type=ast.literal_eval,
                       help="Create dynamic dictionaries")
    group.add_argument('-share_vocab', default=False, type=ast.literal_eval,
                       help="Share source and target vocabulary")

    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    group.add_argument('-src_seq_length', type=int, default=10000,
                       help="Maximum source sequence length")
    group.add_argument('-src_seq_length_trunc', type=int, default=0,
                       help="Truncate source sequence length.")
    group.add_argument('-tgt_seq_length', type=int, default=1000,
                       help="Maximum target sequence length to keep.")
    group.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                       help="Truncate target sequence length.")
    group.add_argument('-lower', action='store_true', help='lowercase datasets')

    # Data processing options
    group = parser.add_argument_group('Random')
    group.add_argument('-shuffle', type=int, default=1,
                       help="Shuffle datasets")
    group.add_argument('-seed', type=int, default=3435,
                       help="Random seed")

    group = parser.add_argument_group('Logging')
    group.add_argument('-report_every', type=int, default=100000,
                       help="Report status every this many sentences")

    # Options most relevant to speech
    group = parser.add_argument_group('Speech')
    group.add_argument('-sample_rate', type=int, default=16000,
                       help="Sample rate.")
    group.add_argument('-window_size', type=float, default=.02,
                       help="Window size for spectrogram in seconds.")
    group.add_argument('-window_stride', type=float, default=.01,
                       help="Window stride for spectrogram in seconds.")
    group.add_argument('-window', default='hamming',
                       help="Window type for spectrogram generation.")


def train_opts(parser):
    # Model loading/saving options

    group = parser.add_argument_group('General')
    group.add_argument('-training_with_val', default=True, type=ast.literal_eval)

    group.add_argument('-data', default='', type=str,
                       help="""Path prefix to the ".train.pt" and
                       ".valid.pt" file path from preprocess.py""")

    group.add_argument('-save_model', default='',
                       help="""Model filename (the model will be saved as
                       <save_model>_epochN_PPL.pt where PPL is the
                       validation perplexity""")
    # GPU
    group.add_argument('-gpuid', default=[0], nargs='+', type=int,
                       help="Use CUDA on the listed devices.")

    group.add_argument('-seed', type=int, default=1234,
                       help="""Random seed used for the experiments
                       reproducibility.""")

    # Init options
    group = parser.add_argument_group('Initialization')
    group.add_argument('-start_epoch', type=int, default=1,
                       help='The epoch from which to start')
    group.add_argument('-param_init', type=float, default=0.1,
                       help="""Parameters are initialized over uniform distribution
                       with support (-param_init, param_init).
                       Use 0 to not use initialization""")
    group.add_argument('-param_init_glorot', default=False, type=ast.literal_eval,
                       help="Init parameters with xavier_uniform. "
                            "Required for transformer.")

    group.add_argument('-train_from', default='',
                       type=str,
                       help="""If training from a checkpoint then this is the
                       path to the pretrained model's state_dict.""")

    # Pretrained word vectors
    group.add_argument('-pre_word_vecs_enc',
                       help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the encoder side.
                       See README for specific formatting instructions.""")
    group.add_argument('-pre_word_vecs_dec',
                       help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the decoder side.
                       See README for specific formatting instructions.""")
    # Fixed word vectors
    group.add_argument('-fix_word_vecs_enc',
                       action='store_true',
                       help="Fix word embeddings on the encoder side.")
    group.add_argument('-fix_word_vecs_dec',
                       action='store_true',
                       help="Fix word embeddings on the encoder side.")
    group.add_argument('-only_ptr', default=False, type=ast.literal_eval)

    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add_argument('-is_sharing_optim', default=True, type=ast.literal_eval)
    group.add_argument('-batch_size', type=int, default=5,
                       help='Maximum batch size for training')
    group.add_argument('-batch_type', default='sents',
                       choices=["sents", "tokens"],
                       help="""Batch grouping for batch_size. Standard
                               is sents. Tokens will do dynamic batching""")
    group.add_argument('-normalization', default='sents',
                       choices=["sents", "tokens"],
                       help='Normalization method of the gradient.')
    group.add_argument('-accum_count', type=int, default=1,
                       help="""Accumulate gradient this many times.
                       Approximately equivalent to updating
                       batch_size * accum_count batches at once.
                       Recommended for Transformer.""")
    group.add_argument('-valid_batch_size', type=int, default=6,
                       help='Maximum batch size for validation')
    group.add_argument('-max_generator_batches', type=int, default=32,
                       help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but
                        uses more memory.""")
    group.add_argument('-epochs', type=int, default=50,
                       help='Number of training epochs')
    group.add_argument('-optim_1', default='adagrad',
                       choices=['sgd', 'adagrad', 'adadelta', 'adam'],
                       help="""Optimization method. for col_sorted""")
    group.add_argument('-optim_2', default='adagrad',
                       choices=['sgd', 'adagrad', 'adadelta', 'adam'],
                       help="""Optimization method. for nmt""")
    group.add_argument('-optim_3', default='adagrad',
                       choices=['sgd', 'adagrad', 'adadelta', 'adam'],
                       help="""Optimization method. for row_sorted""")
    group.add_argument('-adagrad_accumulator_init', type=float, default=0.1,
                       help="""Initializes the accumulator values in adagrad.
                       Mirrors the initial_accumulator_value option
                       in the tensorflow adagrad (use 0.1 for their default).
                       """)
    group.add_argument('-max_grad_norm_1', type=float, default=5,
                       help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to
                       max_grad_norm""")
    group.add_argument('-max_grad_norm_2', type=float, default=5,
                       help="""If the norm of the gradient vector exceeds this,
                           renormalize it to have the norm equals to
                           max_grad_norm""")
    group.add_argument('-max_grad_norm_3', type=float, default=5,
                       help="""If the norm of the gradient vector exceeds this,
                               renormalize it to have the norm equal to
                               max_grad_norm""")
    group.add_argument('-dropout', type=float, default=0.3,
                       help="Dropout probability; applied in LSTM stacks.")
    group.add_argument('-truncated_decoder', type=int, default=100,
                       help="""Truncated bptt.""")
    group.add_argument('-adam_beta1', type=float, default=0.9,
                       help="""The beta1 parameter used by Adam.
                       Almost without exception a value of 0.9 is used in
                       the literature, seemingly giving good results,
                       so we would discourage changing this value from
                       the default without due consideration.""")
    group.add_argument('-adam_beta2', type=float, default=0.999,
                       help="""The beta2 parameter used by Adam.
                       Typically a value of 0.999 is recommended, as this is
                       the value suggested by the original paper describing
                       Adam, and is also the value adopted in other frameworks
                       such as Tensorflow and Kerras, i.e. see:
                       https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
                       https://keras.io/optimizers/ .
                       Whereas recently the paper "Attention is All You Need"
                       suggested a value of 0.98 for beta2, this parameter may
                       not work well for normal models / default
                       baselines.""")
    group.add_argument('-label_smoothing', type=float, default=0.0,
                       help="""Label smoothing value epsilon.
                       Probabilities of all non-true labels
                       will be smoothed by epsilon / (vocab_size - 1).
                       Set to zero to turn off label smoothing.
                       For more detailed information, see:
                       https://arxiv.org/abs/1512.00567""")
    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add_argument('-learning_rate_1', type=float, default=0.15,
                       help="""Starting learning rate.
                       Recommended settings: sgd = 1, adagrad = 0.1,
                       adadelta = 1, adam = 0.001""")
    group.add_argument('-learning_rate_2', type=float, default=0.15,
                       help="""Starting learning rate.
                           Recommended settings: sgd = 1, adagrad = 0.1,
                           adadelta = 1, adam = 0.001""")
    group.add_argument('-learning_rate_decay_1', type=float, default=0.97,
                       help="""If updattre_learning_rate, decay learning rate by
                       this much if (i) perplexity does not decrease on the
                       validation set or (ii) epoch has gone past
                       start_decay_at""")
    group.add_argument('-learning_rate_decay_2', type=float, default=0.97,
                       help="""If update_learning_rate, decay learning rate by
                           this much if (i) perplexity does not decrease on the
                           validation set or (ii) epoch has gone past
                           start_decay_at""")
    group.add_argument('-start_decay_at_1', type=int, default=4,
                       help="""Start decaying every epoch after and including this
                       epoch""")
    group.add_argument('-start_decay_at_2', type=int, default=4,
                       help="""Start decaying every epoch after and including this
                           epoch""")
    group.add_argument('-start_checkpoint_at', type=int, default=6,
                       help="""Start checkpointing every epoch after and including
                       this epoch""")
    group.add_argument('-decay_method', type=str, default="",
                       choices=['noam'], help="Use a custom decay rate.")
    group.add_argument('-decay_every', type=int, default=2)
    group.add_argument('-warmup_steps', type=int, default=4000,
                       help="""Number of warmup steps for custom decay.""")

    group = parser.add_argument_group('Logging')
    group.add_argument('-report_every', type=int, default=50,
                       help="Print stats at this interval.")
    group.add_argument('-exp_host', type=str, default="",
                       help="Send logs to this crayon server.")
    group.add_argument('-exp', type=str, default="",
                       help="Name of the experiment for logging.")
    # Use TensorboardX for visualization during training
    group.add_argument('-tensorboard', default=False, type=ast.literal_eval,
                       help="""Use tensorboardX for visualization during training.
                       Must have the library tensorboardX.""")
    group.add_argument("-tensorboard_log_dir", type=str, default="runs/fix_DD_col_row_0.9",
                       help="""Log directory for Tensorboard.
                       This is also the name of the run.
                       """)

    group = parser.add_argument_group('Speech')
    # Options most relevant to speech
    group.add_argument('-sample_rate', type=int, default=16000,
                       help="Sample rate.")
    group.add_argument('-window_size', type=float, default=.02,
                       help="Window size for spectrogram in seconds.")


def translate_opts(parser):
    group = parser.add_argument_group('Model')
    group.add_argument('-model2',
                       default='',
                       type=str,
                       help='Path to second model .pt file')

    group = parser.add_argument_group('Data')
    group.add_argument('-data_type', default="text",
                       help="Type of the source input. Options: [text|img].")

    group.add_argument('-src',   default='rotowire/inf_src_valid.txt',
                       help="""Source sequence to decode (one line per
                       sequence)""")
    group.add_argument('-src_hist',   default='rotowire/hist_full/inf_src_test_hist_3.txt', type=str,
                       help="""Source sequence to decode (one line per
                       sequence)""")
    group.add_argument('-src_dir',   default="",
                       help='Source directory for image or audio files')
    group.add_argument('-tgt',
                       help='True target sequence (optional)')
    group.add_argument('-output', default='',
                       help="Path to output the predictions")
    group.add_argument('-report_bleu', action='store_true',
                       help="""Report bleu score after translation,
                       call tools/multi-bleu.perl on command line""")
    group.add_argument('-report_rouge', action='store_true',
                       help="""Report rouge 1/2/3/L/SU4 score after translation
                       call tools/test_rouge.py on command line""")

    # Options most relevant to summarization.
    group.add_argument('-dynamic_dict', default=True, type=ast.literal_eval,
                       help="Create dynamic dictionaries")
    group.add_argument('-share_vocab', default=False, type=ast.literal_eval,
                       help="Share source and target vocabulary")

    group = parser.add_argument_group('Beam')
    group.add_argument('-beam_size',  type=int, default=5,
                       help='Beam size')
    group.add_argument('-min_length', type=int, default=150,
                       help='Minimum prediction length')
    group.add_argument('-max_length', type=int, default=850,
                       help='Maximum prediction length.')
    group.add_argument('-max_sent_length', action=DeprecateAction,
                       help="Deprecated, use `-max_length` instead")

    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    group.add_argument('-stepwise_penalty', action='store_true',
                       help="""Apply penalty at every decoding step.
                       Helpful for summary penalty.""")
    group.add_argument('-length_penalty', default='none',
                       choices=['none', 'wu', 'avg'],
                       help="""Length Penalty to use.""")
    group.add_argument('-coverage_penalty', default='none',
                       choices=['none', 'wu', 'summary'],
                       help="""Coverage Penalty to use.""")
    group.add_argument('-alpha', type=float, default=0.,
                       help="""Google NMT length penalty parameter
                        (higher = longer generation)""")
    group.add_argument('-beta', type=float, default=-0.,
                       help="""Coverage penalty parameter""")
    group.add_argument('-replace_unk', action="store_true",
                       help="""Replace the generated UNK tokens with the
                       source token that had highest attention weight. If
                       phrase_table is provided, it will lookup the
                       identified source token and give the corresponding
                       target token. If it is not provided(or the identified
                       source token does not exist in the table) then it
                       will copy the source token""")

    group = parser.add_argument_group('Logging')
    group.add_argument('-verbose', action="store_true",
                       help='Print scores and predictions for each sentence')
    group.add_argument('-attn_debug', action="store_true",
                       help='Print best attn for each word')
    group.add_argument('-dump_beam', type=str, default="",
                       help='File to dump beam information to.')
    group.add_argument('-n_best', type=int, default=1,
                       help="""If verbose is set, will output the n_best
                       decoded sentences""")

    group = parser.add_argument_group('Efficiency')
    group.add_argument('-batch_size', type=int, default=5,
                       help='Batch size')
    group.add_argument('-gpu', type=int, default=0,
                       help="Device to run on")

    # Options most relevant to speech.
    group = parser.add_argument_group('Speech')
    group.add_argument('-sample_rate', type=int, default=16000,
                       help="Sample rate.")
    group.add_argument('-window_size', type=float, default=.02,
                       help='Window size for spectrogram in seconds')
    group.add_argument('-window_stride', type=float, default=.01,
                       help='Window stride for spectrogram in seconds')
    group.add_argument('-window', default='hamming',
                       help='Window type for spectrogram generation')

    group.add_argument('-stage1', action="store_true",
                       help="Stage1 pre process")

def add_md_help_argument(parser):
    parser.add_argument('-md', action=MarkdownHelpAction,
                        help='print Markdown-formatted help text and exit.')


# MARKDOWN boilerplate

# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
class MarkdownHelpFormatter(argparse.HelpFormatter):
    """A really bare-bones argparse help formatter that generates valid markdown.
    This will generate something like:
    usage
    # **section heading**:
    ## **--argument-one**
    ```
    argument-one help text
    ```
    """

    def _format_usage(self, usage, actions, groups, prefix):
        return ""

    def format_help(self):
        print(self._prog)
        self._root_section.heading = '# Options: %s' % self._prog
        return super(MarkdownHelpFormatter, self).format_help()

    def start_section(self, heading):
        super(MarkdownHelpFormatter, self)\
            .start_section('### **%s**' % heading)

    def _format_action(self, action):
        if action.dest == "help" or action.dest == "md":
            return ""
        lines = []
        lines.append('* **-%s %s** ' % (action.dest,
                                        "[%s]" % action.default
                                        if action.default else "[]"))
        if action.help:
            help_text = self._expand_help(action)
            lines.extend(self._split_lines(help_text, 80))
        lines.extend(['', ''])
        return '\n'.join(lines)


class MarkdownHelpAction(argparse.Action):
    def __init__(self, option_strings,
                 dest=argparse.SUPPRESS, default=argparse.SUPPRESS,
                 **kwargs):
        super(MarkdownHelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.formatter_class = MarkdownHelpFormatter
        parser.print_help()
        parser.exit()


class DeprecateAction(argparse.Action):
    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0,
                                              help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.help is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise argparse.ArgumentTypeError(msg)
