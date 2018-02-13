#!/bin/bash

file="nyt_bagrnn_adv_1"
if [ $# -gt 0 ]; then
    file=$1
fi

if [ ! -d ./model/$file ]; then
    mkdir -p ./model/$file
    mkdir -p ./log/$file
    mkdir -p ./log/$file/train
    mkdir -p ./log/$file/test
    mkdir -p ./stats/$file
fi

#rm -r ./model/$file/*
# rm -r ./log/$file/train/*
# rm -r ./log/$file/test/*
#rm -r ./stats/$file/*

CUDA_VISIBLE_DEVICES=0 python3 bag_runner.py --name $file --epoch 40 \
    --lrate 0.001 \
    --model_dir ./model/$file --log ./log/$file --eval_dir ./stats/$file \
    --bag_num 50 \
    --vocab_size 80000 \
    --L 145 \
    --entity_dim 3 \
    --enc_dim 150 \
    --cat_n 5 \
    --cell_type gru \
    --lrate_decay 0 \
    --report_rate 0.2 \
    --seed 57 \
    --clip_grad 10 \
    --gpu_usage 0.9 \
    --dropout 0.5 \
    --adv_eps 0.02 \
    --dataset nyt --cat_n 58 \
    --max_eval_rel 3000
