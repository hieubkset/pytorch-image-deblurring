import os.path
import random
import torch.utils.data as data
from utils import *
from PIL import Image
from torchvision import transforms
import numpy as np


def augment(img_input, img_target):
    degree = random.choice([0, 90, 180, 270])
    img_input = transforms.functional.rotate(img_input, degree)
    img_target = transforms.functional.rotate(img_target, degree)

    # color augmentation
    img_input = transforms.functional.adjust_gamma(img_input, 1)
    img_target = transforms.functional.adjust_gamma(img_target, 1)
    sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
    img_input = transforms.functional.adjust_saturation(img_input, sat_factor)
    img_target = transforms.functional.adjust_saturation(img_target, sat_factor)

    return img_input, img_target


def getPatch(img_input, img_target, path_size):
    w, h = img_input.size
    p = path_size
    x = random.randrange(0, w - p + 1)
    y = random.randrange(0, h - p + 1)
    img_input = img_input.crop((x, y, x + p, y + p))
    img_target = img_target.crop((x, y, x + p, y + p))
    return img_input, img_target


class Gopro(data.Dataset):
    def __init__(self, data_dir, patch_size=256, is_train=False, multi=True):
        super(Gopro, self).__init__()
        self.is_train = is_train
        self.patch_size = patch_size
        self.multi = multi

        self.sharp_file_paths = []

        sub_folders = os.listdir(data_dir)

        for folder_name in sub_folders:
            sharp_sub_folder = os.path.join(data_dir, folder_name, 'sharp')
            sharp_file_names = os.listdir(sharp_sub_folder)

            for file_name in sharp_file_names:
                sharp_file_path = os.path.join(sharp_sub_folder, file_name)
                self.sharp_file_paths.append(sharp_file_path)

        self.n_samples = len(self.sharp_file_paths)

    def get_img_pair(self, idx):
        sharp_file_path = self.sharp_file_paths[idx]
        blur_file_path = sharp_file_path.replace("sharp", "blur")

        img_input = Image.open(blur_file_path).convert('RGB')
        img_target = Image.open(sharp_file_path).convert('RGB')

        return img_input, img_target

    def __getitem__(self, idx):
        img_input, img_target = self.get_img_pair(idx)

        if self.is_train:
            img_input, img_target = getPatch(img_input, img_target, self.patch_size)
            img_input, img_target = augment(img_input, img_target)

        input_b1 = transforms.ToTensor()(img_input)
        target_s1 = transforms.ToTensor()(img_target)

        H = input_b1.size()[1]
        W = input_b1.size()[2]

        if self.multi:
            input_b1 = transforms.ToPILImage()(input_b1)
            target_s1 = transforms.ToPILImage()(target_s1)

            input_b2 = transforms.ToTensor()(transforms.Resize([int(H / 2), int(W / 2)])(input_b1))
            input_b3 = transforms.ToTensor()(transforms.Resize([int(H / 4), int(W / 4)])(input_b1))

            if self.is_train:
                target_s2 = transforms.ToTensor()(transforms.Resize([int(H / 2), int(W / 2)])(target_s1))
                target_s3 = transforms.ToTensor()(transforms.Resize([int(H / 4), int(W / 4)])(target_s1))
            else:
                target_s2 = []
                target_s3 = []

            input_b1 = transforms.ToTensor()(input_b1)
            target_s1 = transforms.ToTensor()(target_s1)
            return {'input_b1': input_b1, 'input_b2': input_b2, 'input_b3': input_b3,
                    'target_s1': target_s1, 'target_s2': target_s2, 'target_s3': target_s3}
        else:
            return {'input_b1': input_b1, 'target_s1': target_s1}

    def __len__(self):
        return self.n_samples
