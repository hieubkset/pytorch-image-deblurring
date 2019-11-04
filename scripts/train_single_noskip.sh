#!/usr/bin/env bash
# singlescale and no skip connection
#python main.py --gpu 2 --exp_name single_noskip
python main.py --gpu 5 --lr 1e-5 --exp_name single_noskip_lre4 --finetuning