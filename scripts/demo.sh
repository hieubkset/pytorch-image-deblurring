#!/usr/bin/env bash
# generate images by using provided models
python demo.py --gpu 0 --train_dir pretrained --exp_name multi_skip --image "dataset/test/GOPR0384_11_00/blur/000001.png" "dataset/test/GOPR0384_11_05/blur/004001.png" "dataset/test/GOPR0385_11_01/blur/003011.png"
#python demo.py --gpu 1 --train_dir pretrained --exp_name multi_noskip --image "dataset/test/GOPR0384_11_00/blur/000001.png" "dataset/test/GOPR0384_11_05/blur/004001.png" "dataset/test/GOPR0385_11_01/blur/003011.png"
#python demo.py --gpu 2 --train_dir pretrained --exp_name single_skip --image "dataset/test/GOPR0384_11_00/blur/000001.png" "dataset/test/GOPR0384_11_05/blur/004001.png" "dataset/test/GOPR0385_11_01/blur/003011.png"
#python demo.py --gpu 3 --train_dir pretrained --exp_name single_noskip --image "dataset/test/GOPR0384_11_00/blur/000001.png" "dataset/test/GOPR0384_11_05/blur/004001.png" "dataset/test/GOPR0385_11_01/blur/003011.png"
# generate images by your trained model
#python demo.py --gpu 0 --train_dir result --exp_name multi_skip --image "dataset/test/GOPR0384_11_00/blur/000001.png" "dataset/test/GOPR0384_11_05/blur/004001.png" "dataset/test/GOPR0385_11_01/blur/003011.png"
#python demo.py --gpu 1 --train_dir pretrained --exp_name multi_noskip --image "dataset/test/GOPR0384_11_00/blur/000001.png" "dataset/test/GOPR0384_11_05/blur/004001.png" "dataset/test/GOPR0385_11_01/blur/003011.png"
#python demo.py --gpu 2 --train_dir pretrained --exp_name single_skip --image "dataset/test/GOPR0384_11_00/blur/000001.png" "dataset/test/GOPR0384_11_05/blur/004001.png" "dataset/test/GOPR0385_11_01/blur/003011.png"
#python demo.py --gpu 3 --train_dir pretrained --exp_name single_noskip --image "dataset/test/GOPR0384_11_00/blur/000001.png" "dataset/test/GOPR0384_11_05/blur/004001.png" "dataset/test/GOPR0385_11_01/blur/003011.png"
