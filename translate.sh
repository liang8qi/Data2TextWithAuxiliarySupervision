#!/bin/bash

BASE=dataset/rotowire
RESULT_DIR=results
PRED_FILE_NAME=model_pred_rw_test.txt
MODEL_NAME=$BASE/gen_model/*.pt
DEVICE=0
mkdir $RESULT_DIR

CUDA_VISIBLE_DEVICES=$DEVICE python translate.py \
-share_vocab False \
-model $MODEL_NAME \
-src $BASE/inf_src_test.txt \
-output $RESULT_DIR/$PRED_FILE_NAME \
-beam_size 5 \
-min_length 150 \
-max_length 850 \
-batch_size 5