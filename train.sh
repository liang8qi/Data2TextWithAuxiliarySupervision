BASE=dataset/rotowire
SAVE_MODEL=$BASE/gen_model/
DATA_NAME=rw_DA
LOG_DIR=logs
RESULT_DIR=results/
MODEL_NAME=model
DEVICE=0
mkdir $SAVE_MODEL
mkdir $LOG_DIR
mkdir $RESULT_DIR

CUDA_VISIBLE_DEVICES=$DEVICE python train.py -hier_meta $BASE/hier_meta.json \
-data $BASE/$DATA_NAME/roto \
-share_embeddings False \
-param_init_glorot False \
-no_fusion False \
-row_rep_type gnn \
-gnn_type gategat \
-gat_attn_type dot \
-gnn_share False \
-gnn_n_layers 2 \
-gat_n_heads 2 \
-gat_dropout 0.3 \
-gat_is_self False \
-gnn_activation_type elu \
-gat_residual True \
-gat_leaky_relu_alpha 0.2 \
-word_vec_size 600 \
-hier_rnn_size 600 \
-feat_vec_size 100 \
-row_aggregation attention \
-row_agg_is_param False \
-aggr_att_type mlp \
-decoder_type2 rnn \
-dec_layers2 2 \
-rnn_size 600 \
-enable_number_ranking True \
-enable_importance_ranking True \
-alpha_nr 0.9 \
-alpha_ir 0.25 \
-seed 1234 \
-training_with_val True \
-batch_size 5 \
-valid_batch_size 6 \
-max_generator_batches 32 \
-truncated_decoder 100 \
-epochs 50 \
-optim_2 adagrad \
-adagrad_accumulator_init 0.1 \
-learning_rate_2 0.15 \
-learning_rate_decay_2 0.97 \
-start_decay_at_2 4 \
-decay_every 1 \
-max_grad_norm_2 5 \
-dropout 0.3 \
-start_checkpoint_at 6 \
-report_every 50 \
-tensorboard False \
-tensorboard_log_dir $LOG_DIR/test \
-save_model $SAVE_MODEL/$MODEL_NAME/roto