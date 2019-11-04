import os
import os.path
import torch
import sys
import numpy as np
from math import log10
from ssim import SSIM
from tensorboardX import SummaryWriter


def tensor_to_rgb(img_input):
    output = img_input.cpu()
    output = output.data.squeeze(0)

    output = output.numpy()
    output *= 255.0
    output = output.clip(0, 255)

    return output


def compute_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    psnr = 10 * log10(255 * 255 / (mse + 10 ** (-10)))
    return psnr


class SaveData():
    def __init__(self, save_dir, exp_name, finetuning):
        self.save_dir = os.path.join(save_dir, exp_name)

        if not finetuning:
            if os.path.exists(self.save_dir):
                os.system('rm -rf ' + self.save_dir)
            os.makedirs(self.save_dir)
        else:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)

        self.logFile = open(self.save_dir + '/log.txt', 'a')

        save_dir_tensorboard = os.path.join(self.save_dir, 'logs')
        if not os.path.exists(save_dir_tensorboard):
            os.makedirs(save_dir_tensorboard)
        self.writer = SummaryWriter(save_dir_tensorboard)


    def save_params(self, args):
        with open(self.save_dir + '/params.txt', 'w') as params_file:
            params_file.write(str(args.__dict__) + "\n\n")


    def save_model(self, model, epoch):
        torch.save(model.state_dict(), self.save_dir_model + '/model_lastest.pt')
        torch.save(model.state_dict(), self.save_dir_model + '/model_' + str(epoch) + '.pt')
        torch.save(model, self.save_dir_model + '/model_obj.pt')
        torch.save(epoch, self.save_dir_model + '/last_epoch.pt')

    def save_log(self, log):
        sys.stdout.flush()
        self.logFile.write(log + '\n')
        self.logFile.flush()

    def load_model(self, model):
        model.load_state_dict(torch.load(self.save_dir_model + '/model_lastest.pt'))
        last_epoch = torch.load(self.save_dir_model + '/last_epoch.pt')
        print("Load mode_status from {}/model_lastest.pt, epoch: {}".format(self.save_dir_model, last_epoch))
        return model, last_epoch

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
