import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from scipy import ndimage
from configs.config import get_cfg_defaults

config = get_cfg_defaults()
T_ori, H_ori, W_ori, input_img_size= config.TEST.input_D, config.TEST.input_H, config.TEST.input_W, config.TRAIN.input_size

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        H = random.randint(0, H_ori - input_img_size)
        W = random.randint(0, W_ori - input_img_size)
        T = random.randint(0, T_ori - input_img_size)

        image = image[H: H + input_img_size, W: W + input_img_size, T: T + input_img_size, ...]
        label = label[..., H: H + input_img_size, W: W + input_img_size, T: T + input_img_size]

        return {'image': image, 'label': label}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']

        image = np.ascontiguousarray(image.transpose(3, 2, 0, 1)) # HWDC -> CDHW    BCDWH
        label = sample['label']
        label = np.ascontiguousarray(label.transpose(2, 0, 1))

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        return {'image': image, 'label': label}


def transform(sample):
    trans = transforms.Compose([
        Random_Crop(),
        Random_Flip(),
        Random_intencity_shift(),
        ToTensor()
    ])

    return trans(sample)


def transform_valid(sample):
    trans = transforms.Compose([
        ToTensor()
    ])

    return trans(sample)


class BraTS2019(Dataset):
    def __init__(self, list_file, root='', mode='train'):
        self.lines = []
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line, name + '_')
                paths.append(path)
                self.lines.append(line)
        self.mode = mode
        self.names = names
        self.paths = paths

    def __getitem__(self, item):
        path = self.paths[item]
        if self.mode == 'train':
            image, label = pkload(path + 'data_f32b0.pkl') #2019
            
            sample = {'image': image, 'label': label}
            sample = transform(sample)
            return sample['image'], sample['label']

        elif self.mode == 'valid':
            image, label = pkload(path + 'data_f32b0.pkl') #2019
            sample = {'image': image, 'label': label}
            sample = transform_valid(sample)
            return sample['image'], sample['label']
            
        else: # test
            image = pkload(path + 'data_f32b0.pkl') #2019

            image = np.ascontiguousarray(image.transpose(3, 2, 0, 1)) # HWDC -> CDHW    BCDWH

            image = torch.from_numpy(image).float()
            return image

    def __len__(self):
        return len(self.paths)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


class BraTS2020(Dataset):
    def __init__(self, list_file, root='', mode='train'):
        self.lines = []
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line, name + '_')
                paths.append(path)
                self.lines.append(line)
        self.mode = mode
        self.names = names
        self.paths = paths

    def __getitem__(self, item):
        path = self.paths[item]
        if self.mode == 'train':
            image, label = pkload(path + 'data_f32.pkl') #2020
            
            sample = {'image': image, 'label': label}
            sample = transform(sample)
            return sample['image'], sample['label']

        elif self.mode == 'valid':
            image, label = pkload(path + 'data_f32.pkl') #2020

            sample = {'image': image, 'label': label}
            sample = transform_valid(sample)
            return sample['image'], sample['label']
            
        else: # test
            image = pkload(path + 'data_f32.pkl') #2020

            image = np.ascontiguousarray(image.transpose(3, 2, 0, 1)) # HWDC -> CDHW    BCDWH

            image = torch.from_numpy(image).float()
            return image

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


