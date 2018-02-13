#!/usr/bin/env bash

file="nyt_pcnn_sent_adv_4"
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

CUDA_VISIBLE_DEVICES=0 python3 bag_runner.py --name $file --epoch 50 \
    --lrate 0.001 \
    --model_dir ./model/$file --log ./log/$file --eval_dir ./stats/$file \
    --bag_num 50 \
    --vocab_size 80000 \
    --L 145 \
    --entity_dim 5 \
    --enc_dim 230 \
    --cat_n 5 \
    --no-embed-dropout \
    --cell_type pcnn \
    --lrate_decay 0 \
    --report_rate 0.2 \
    --seed 57 \
    --clip_grad 10 \
    --gpu_usage 0.9 \
    --dropout 0.5 \
    --adv_eps 0.01 \
    --dataset nyt --cat_n 58 \
    --max_eval_rel 3000 \
    --sentence_eps 0.075
#    --warmstart ./model/nyt_pcnn_pos_adv/nyt_pcnn_pos_adv_ep40
