# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
from PIL import Image
import cv2
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from skimage import transform
from torchvision import transforms
import random

class DisasterDataset(Dataset):
    def __init__(self, data_dir, data_dir_ls, data_mean_stddev, transform:bool, normalize:bool):
        
        self.data_dir = data_dir
        self.dataset_sub_dir = data_dir_ls
        self.data_mean_stddev = data_mean_stddev
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.dataset_sub_dir)

    def __getitem__(self, i):
        
        imgs_dir = self.data_dir + self.dataset_sub_dir[i].replace('labels', 'images')
        masks_dir = self.data_dir + self.dataset_sub_dir[i].replace('labels', 'targets_border2')

        idx = imgs_dir

        img_suffix = '_' + imgs_dir.split('_')[-1]
        mask_suffix = '_' + masks_dir.split('_')[-1]


        pre_img_tile_name = imgs_dir[0:-1*(len(img_suffix))] + '_pre_disaster'    
        pre_img_file_name = imgs_dir[0:-1*(len(img_suffix))] + '_pre_disaster' + img_suffix
        pre_img_file = glob(pre_img_file_name + '.*')

        mask_file_name = masks_dir[0:-1*(len(mask_suffix))] + '_pre_disaster_b2' + mask_suffix
        mask_file = glob(mask_file_name + '.*')

        post_img_tile_name = pre_img_tile_name.replace('pre', 'post')
        post_img_file_name = pre_img_file_name.replace('pre', 'post')
        post_img_file = glob(post_img_file_name + '.*')

        damage_class_file_name = mask_file_name.replace('pre', 'post')
        damage_class_file = glob(damage_class_file_name + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file_name}'
        assert len(pre_img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {pre_img_file_name}'
        assert len(post_img_file) == 1, \
            f'Either no post disaster image or multiple images found for the ID {idx}: {post_img_file_name}'
        assert len(damage_class_file) == 1, \
            f'Either no damage class image or multiple images found for the ID {idx}: {damage_class_file_name}'

        mask = cv2.imread(mask_file[0], cv2.IMREAD_GRAYSCALE)
        pre_img = cv2.imread(pre_img_file[0])
        post_img = cv2.imread(post_img_file[0])
        damage_class = cv2.imread(damage_class_file[0], cv2.IMREAD_GRAYSCALE)

        assert pre_img.shape[0] == mask.shape[0], \
            f'Image and building mask {idx} should be the same size, but are {pre_img.shape} and {mask.shape}'
        assert mask.size == damage_class.size, \
            f'Image and damage classes mask {idx} should be the same size, but are {mask.size} and {damage_class.size}'
        assert pre_img.size == post_img.size, \
            f'Pre_ & _post disaster Images {idx} should be the same size, but are {pre_img.size} and {post_img.size}'

        data = {'pre_image': pre_img, 'post_image': post_img, 'building_mask': mask, 'damage_mask': damage_class, 'pre_img_tile_name': pre_img_tile_name}
        
        return data