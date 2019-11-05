EE838A - Homework 2 - Image Deblurring

This is an instruction for using this source code. The source code is tested on Ubuntu 16.04 using GPU Titan XP.
Note that one GPU is required to run the code. Otherwise, you have to change code by yourself a little bit.

1. Dependencies:
Python 3.5 or 3.6 are recommended.
tqdm==4.19.9
numpy==1.17.3
tensorflow==1.9.0
tensorboardX==1.9
torch==1.0.0
Pillow==6.1.0
torchvision==0.2.2

2. Environment
I recommend using virtualenv for making an environment. If you using virtualenv, run the following commands in the
root of source code folder.
```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

3. Dataset
By default, you should have images on dataset/train and dataset/valid folders.
If you change where stores dataset, it requires to change .sh files in scripts folder.
You may find that 'ln -s' command is useful for preparing data.

4. Demo
I provide pretrained models in pretrained folder. You can generate deblurred images by running the following command:
```
sh scripts/demo.sh
```
The above command may be failed due to the difference of line separator between Window and Ubuntu. You may have to use
a direct command:
```
python demo.py --gpu 0 --train_dir pretrained --exp_name multi_skip --image "dataset/test/GOPR0384_11_00/blur/000001.png" "dataset/test/GOPR0384_11_05/blur/004001.png" "dataset/test/GOPR0385_11_01/blur/003011.png"
```
For using other models, you should uncommend lines in scripts/demo.sh file.

5. Training
Run the following command
```
sh scripts/train.sh
```
For training other models, you should uncommend lines in scripts/train.sh file.

6. Testing
```
sh scripts/test.sh
```
For testing other models, you should uncommend lines in scripts/test.sh file.