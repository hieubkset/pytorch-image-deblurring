#!/usr/bin/env bash
# multiscale and no skip connection
#python main.py --gpu 1 --multi --exp_name multi_noskip

python main.py --gpu 4 --multi --lr 1e-4 --exp_name multi_noskip_lre4 --finetuning

