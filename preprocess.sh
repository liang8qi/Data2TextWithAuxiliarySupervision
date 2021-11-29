#!/bin/bash

Dataset_Path=dataset/rotowire/
Saved_Name=rw_DA

python preprocess.py \
-share_vocab False \
-cal_indices True \
-save_data $Dataset_Path/$Saved_Name/roto \
-process_hier_meta $Dataset_Path/hier_meta.json \
-train_src $Dataset_Path/src_train.txt \
-train_tgt $Dataset_Path/tgt_train.txt \
-valid_src $Dataset_Path/src_valid.txt \
-valid_tgt $Dataset_Path/tgt_valid.txt \
-train_ptr $Dataset_Path/enc-dec-train-roto-ptrs.txt