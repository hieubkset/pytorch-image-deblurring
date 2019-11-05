#!/usr/bin/env bash
# test your models (save_dir default is result)
python test.py --gpu 0 --exp_name multi_skip --padding 0
#python test.py --gpu 1 --exp_name multi_noskip --padding 0
#python test.py --gpu 2 --exp_name single_skip --padding 0
#python test.py --gpu 3 --exp_name single_noskip --padding 0
##
# test pretrained models
#python test.py  --gpu 0 --save_dir pretrained --exp_name multi_skip --padding 0
#python test.py --gpu 1 --save_dir pretrained --exp_name multi_noskip --padding 0
#python test.py --gpu 2 --save_dir pretrained --exp_name single_skip --padding 0
#python test.py --gpu 3 --save_dir pretrained --exp_name single_noskip --padding 0
