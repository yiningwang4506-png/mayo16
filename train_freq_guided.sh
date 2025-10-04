#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
 --model_name corediff \
 --run_name freq_guided_mayo2016 \
 --batch_size 4 \
 --max_iter 150000 \
 --train_dataset mayo_2016 \
 --test_dataset mayo_2016 \
 --test_id 9 \
 --context \
 --only_adjust_two_step \
 --dose 25 \
 --save_freq 2500 \
 --init_lr 2e-4 \
 --T 10
