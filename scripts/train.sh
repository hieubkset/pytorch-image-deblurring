#!/usr/bin/env bash
# multiscale and skip connection
python main.py --gpu 0 --multi --skip --exp_name multi_skip
# multiscale and no skip connection
#python main.py --gpu 1 --multi --exp_name multi_noskip
# singlescale and no skip connection
#python main.py --gpu 2 --skip --exp_name single_skip
# singlescale and no skip connection
#python main.py --gpu 3 --exp_name single_noskip