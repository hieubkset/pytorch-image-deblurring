#!/usr/bin/env bash
# singlescale and no skip connection
#python main.py --gpu 3 --skip --exp_name single_skip

python main.py --gpu 7 --skip --lr 1e-4 --exp_name single_skip_lre4 --finetuning
