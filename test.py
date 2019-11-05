import os
import argparse
import ast
from model import model
from data import Gopro
import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable
from utils import *
from ssim import ssim as compare_ssim
from msssim import msssim as compare_mssim
from utils import compute_psnr as compare_psnr

parser = argparse.ArgumentParser(description='image-deblurring')

parser.add_argument('--data_dir', type=str, default='dataset/test', help='dataset directory')
parser.add_argument('--save_dir', default='./result', help='data save directory')
parser.add_argument('--exp_name', default='multi_skip', help='model to select')
parser.add_argument('--gpu', type=int, required=True, help='gpu index')
parser.add_argument('--n_threads', type=int, default=8, help='number of threads for data loading')
parser.add_argument('--save', action='store_true', help='Save deblurred image')
parser.add_argument('--padding', type=int, default=8, help='padding for computing scores')

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


def get_dataset(data_dir, n_threads=8, multi=False):
    dataset = Gopro(data_dir, is_train=False, multi=multi)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(n_threads))
    return dataloader


def load_paras(args):
    params_file_path = os.path.join(args.save_dir, args.exp_name, "params.txt")
    with open(params_file_path, 'r') as f:
        str_params = f.read().strip()
    return ast.literal_eval(str_params)


def test(args):
    params = load_paras(args)
    if params['multi']:
        my_model = model.MultiScaleNet(n_feats=params['n_feats'], n_resblocks=params['n_resblocks'],
                                       is_skip=params['skip'])
    else:
        my_model = model.SingleScaleNet(n_feats=params['n_feats'], n_resblocks=params['n_resblocks'],
                                        is_skip=params['skip'])
    my_model.cuda()
    my_model.load_state_dict(torch.load(os.path.join(args.save_dir, args.exp_name, 'model', 'model_lastest.pt')))
    my_model.eval()

    dataloader = get_dataset(args.data_dir, n_threads=args.n_threads, multi=params['multi'])

    if args.save:
        output_dir = os.path.join(args.save_dir, args.exp_name, 'test_output')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    log_file = open(os.path.join(args.save_dir, args.exp_name, 'test_logs.txt'), 'w')

    total_psnr, total_ssim, total_mssim, cnt = 0, 0, 0, 0
    for batch, images in enumerate(dataloader):
        cnt += 1
        with torch.no_grad():
            input_b1 = Variable(images['input_b1'].cuda())
            target_s1 = Variable(images['target_s1'].cuda())

            if params['multi']:
                input_b2 = Variable(images['input_b2'].cuda())
                input_b3 = Variable(images['input_b3'].cuda())
                output_l1, _, _ = my_model((input_b1, input_b2, input_b3))
            else:
                output_l1 = my_model(input_b1)

        output_l1 = tensor_to_rgb(output_l1)
        target_s1 = tensor_to_rgb(target_s1)

        p = args.padding
        if p != 0:
            img1 = output_l1[:, p:-p, p:-p].squeeze()
            img2 = target_s1[:, p:-p, p:-p].squeeze()
        else:
            img1 = output_l1.squeeze()
            img2 = target_s1.squeeze()

        with torch.no_grad():
            mssim = compare_mssim(torch.from_numpy(img1[None]).cuda(),
                                  torch.from_numpy(img2[None]).cuda()).cpu().numpy()
            ssim = compare_ssim(torch.from_numpy(img1[None] / 255.0).cuda(),
                                torch.from_numpy(img2[None] / 255.0).cuda()).cpu().numpy()
        psnr = compare_psnr(img1, img2)

        total_psnr += psnr
        total_ssim += ssim
        total_mssim += mssim

        if args.save:
            out = Image.fromarray(np.uint8(output_l1.transpose(1, 2, 0)), mode='RGB')  # output of SRCNN
            out.save(os.path.join(output_dir, 'DB_%04d.png' % (cnt)))

        log = 'Image %04d - PSNR %.2f - SSIM %.4f - MSSIM %.4f' % (cnt, psnr, ssim, mssim)
        print(log)
        log_file.write(log + "\n")

    avg_psnr = total_psnr / (batch + 1)
    avg_ssim = total_ssim / (batch + 1)
    avg_mssim = total_mssim / (batch + 1)
    log = 'Average - PSNR %.2f dB - SSIM %.4f - MSSIM %.4f' % (avg_psnr, avg_ssim, avg_mssim)
    print(log)
    log_file.write(log + "\n")
    log_file.close()

    if args.save:
        print('%04d images save at %s' % (cnt, output_dir))


if __name__ == '__main__':
    test(args)
