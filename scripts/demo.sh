#!/usr/bin/env bash
#python demo.py --gpu 0 --exp_name multi_skip --image "dataset/test/GOPR0384_11_00/blur/000001.png" "dataset/test/GOPR0384_11_05/blur/004001.png" "dataset/test/GOPR0385_11_01/blur/003011.png"
#python demo.py --gpu 0 --exp_name multi_noskip --image "dataset/test/GOPR0384_11_00/blur/000001.png" "dataset/test/GOPR0384_11_05/blur/004001.png" "dataset/test/GOPR0385_11_01/blur/003011.png"
#python demo.py --gpu 0 --exp_name single_skip --image "dataset/test/GOPR0384_11_00/blur/000001.png" "dataset/test/GOPR0384_11_05/blur/004001.png" "dataset/test/GOPR0385_11_01/blur/003011.png"
python demo.py --gpu 0 --exp_name single_noskip --image "dataset/test/GOPR0384_11_00/blur/000001.png" "dataset/test/GOPR0384_11_05/blur/004001.png" "dataset/test/GOPR0385_11_01/blur/003011.png"