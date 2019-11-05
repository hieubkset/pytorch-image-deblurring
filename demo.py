import os
import argparse
import ast
from model import model
from data import Gopro
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.autograd import Variable
from utils import *

parser = argparse.ArgumentParser(description='image-deblurring')

parser.add_argument('--train_dir', default='./result', help='data save directory')
parser.add_argument('--output_dir', default='demo', help='data save directory')
parser.add_argument('--exp_name', default='Net1', help='model to select')
parser.add_argument('--gpu', type=int, required=True, help='gpu index')
parser.add_argument('--image', nargs='+', required=True, help='image to deblur')

args = parser.parse_args()

if args.gpu == 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif args.gpu == 1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


def get_dataset(data_dir, patch_size=None, batch_size=1, n_threads=8, is_train=False, multi=False):
    dataset = Gopro(data_dir, patch_size=patch_size, is_train=is_train, multi=multi)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             drop_last=True, shuffle=is_train, num_workers=int(n_threads))
    return dataloader


def load_paras(args):
    params_file_path = os.path.join(args.train_dir, args.exp_name, "params.txt")
    with open(params_file_path, 'r') as f:
        str_params = f.read().strip()
    return ast.literal_eval(str_params)


def load_images(blur_img_path, multi):
    target_s1 = None
    sharp_img_path = blur_img_path.replace('blur', 'sharp')
    if os.path.exists(sharp_img_path):
        img_target = Image.open(sharp_img_path).convert('RGB')
        target_s1 = transforms.ToTensor()(img_target)

    img_input = Image.open(blur_img_path).convert('RGB')
    input_b1 = transforms.ToTensor()(img_input)

    if multi:
        H = input_b1.size()[1]
        W = input_b1.size()[2]

        input_b1 = transforms.ToPILImage()(input_b1)

        input_b2 = transforms.ToTensor()(transforms.Resize([int(H / 2), int(W / 2)])(input_b1))
        input_b3 = transforms.ToTensor()(transforms.Resize([int(H / 4), int(W / 4)])(input_b1))

        input_b1 = transforms.ToTensor()(input_b1)
        return {'input_b1': input_b1[None], 'input_b2': input_b2[None], 'input_b3': input_b3[None],
                'target_s1': target_s1[None]}
    else:
        return {'input_b1': input_b1[None], 'target_s1': target_s1[None]}


def test(args):
    params = load_paras(args)
    if params['multi']:
        my_model = model.MultiScaleNet(n_feats=params['n_feats'], n_resblocks=params['n_resblocks'],
                                       is_skip=params['skip'])
    else:
        my_model = model.SingleScaleNet(n_feats=params['n_feats'], n_resblocks=params['n_resblocks'],
                                        is_skip=params['skip'])
    my_model.cuda()
    my_model.load_state_dict(torch.load(os.path.join(args.train_dir, args.exp_name, 'model', 'model_lastest.pt')))
    my_model.eval()

    output_dir = os.path.join(args.train_dir, args.exp_name, args.output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for img_path in args.image:
        with torch.no_grad():
            images = load_images(img_path, params['multi'])
            input_b1 = Variable(images['input_b1'].cuda())
            target_s1 = Variable(images['target_s1'].cuda())

            if params['multi']:
                input_b2 = Variable(images['input_b2'].cuda())
                input_b3 = Variable(images['input_b3'].cuda())
                output_l1, _, _ = my_model((input_b1, input_b2, input_b3))
            else:
                output_l1 = my_model(input_b1)

        output_l1 = tensor_to_rgb(output_l1)
        output_l1 = output_l1.transpose(1, 2, 0)
        if target_s1 is not None:
            target_s1 = tensor_to_rgb(target_s1)
            target_s1 = target_s1.transpose(1, 2, 0)
            psnr = compute_psnr(target_s1, output_l1)
            print('Image %s psnr %.2f dB' % (os.path.basename(img_path), psnr))

        out = Image.fromarray(np.uint8(output_l1), mode='RGB')
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        out.save(output_path)
        print('One image saved at ' + output_path)


if __name__ == '__main__':
    test(args)
