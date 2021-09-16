# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
from PIL import Image
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
import os
class DisasterDataset(Dataset):
    def __init__(self, data_dir, data_dir_ls, data_mean_stddev, transform:bool, normalize:bool):
        
        self.data_dir = data_dir
        self.dataset_sub_dir = data_dir_ls
        self.data_mean_stddev = data_mean_stddev
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.dataset_sub_dir)
    
    @classmethod
    def apply_transform(self, mask, pre_img, post_img, damage_class):
        '''
        apply tranformation functions on PIL images 
        '''
        if random.random() > 0.5:
            # Resize
            img_h = pre_img.size[0]
            img_w = pre_img.size[1]
            
            resize = transforms.Resize(size=(int(round(1.016*img_h)), int(round(1.016*img_w))))
            mask = resize(mask)
            pre_img = resize(pre_img)
            post_img = resize(post_img)
            damage_class = resize(damage_class)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(pre_img, output_size=(img_h, img_w))
            mask = TF.crop(mask, i, j, h, w)
            pre_img = TF.crop(pre_img, i, j, h, w)
            post_img = TF.crop(post_img, i, j, h, w)
            damage_class = TF.crop(damage_class, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            mask = TF.hflip(mask)
            pre_img = TF.hflip(pre_img)
            post_img = TF.hflip(post_img)
            damage_class = TF.hflip(damage_class)

        # Random vertical flipping
        if random.random() > 0.5:
            mask = TF.vflip(mask)
            pre_img = TF.vflip(pre_img)
            post_img = TF.vflip(post_img)
            damage_class = TF.vflip(damage_class)

        return mask, pre_img, post_img, damage_class

    def __getitem__(self, i):
        
        imgs_dir = os.path.join(self.data_dir ,self.dataset_sub_dir[i].replace('labels', 'images'))
        imgs_dir_tile = self.dataset_sub_dir[i].replace('labels', 'images')
        masks_dir = os.path.join(self.data_dir, self.dataset_sub_dir[i].replace('labels', 'targets_border2'))
        preds_dir = os.path.join(self.data_dir ,self.dataset_sub_dir[i].replace('labels', 'predictions'))

        idx = imgs_dir

        img_suffix = '_' + imgs_dir.split('_')[-1]
        img_suffix_tile = '_' + imgs_dir_tile.split('_')[-1]
        mask_suffix = '_' + masks_dir.split('_')[-1]

        pre_img_tile_name = imgs_dir_tile[0:-1*(len(img_suffix_tile))] + '_pre_disaster'
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

        mask = Image.open(mask_file[0])
        pre_img = Image.open(pre_img_file[0])
        post_img = Image.open(post_img_file[0])
        damage_class = Image.open(damage_class_file[0])

        assert pre_img.size == mask.size, \
            f'Image and building mask {idx} should be the same size, but are {pre_img.size} and {mask.size}'
        assert pre_img.size == damage_class.size, \
            f'Image and damage classes mask {idx} should be the same size, but are {pre_img.size} and {damage_class.size}'
        assert pre_img.size == post_img.size, \
            f'Pre_ & _post disaster Images {idx} should be the same size, but are {pre_img.size} and {post_img.size}'

        if self.transform is True:
            mask, pre_img, post_img, damage_class = self.apply_transform(mask, pre_img, post_img, damage_class)
        
        # copy original image for viz
        pre_img_orig = pre_img
        post_img_orig = post_img

        if self.normalize is True:
            # normalize the images based on a tilewise mean & std dev --> pre_
            mean_pre = self.data_mean_stddev[pre_img_tile_name][0]
            stddev_pre = self.data_mean_stddev[pre_img_tile_name][1]
            norm_pre = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean_pre, std=stddev_pre)
                    ])
            pre_img = norm_pre(np.array(pre_img).astype(dtype='float64')/255.0)

            # normalize the images based on a tilewise mean & std dev --> post_
            mean_post = self.data_mean_stddev[post_img_tile_name][0]
            stddev_post = self.data_mean_stddev[post_img_tile_name][1]
            norm_post = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean_post, std=stddev_post)
                    ])
            post_img = norm_post(np.array(post_img).astype(dtype='float64')/255.0)

        # convert eveything to arrays
        pre_img = np.array(pre_img)
        post_img = np.array(post_img)
        mask = np.array(mask) 
        damage_class = np.array(damage_class)

        # replace non-classified pixels with background
        damage_class = np.where(damage_class==5, 0, damage_class)
        
        return {'pre_image': torch.from_numpy(pre_img).type(torch.FloatTensor), 'post_image': torch.from_numpy(post_img).type(torch.FloatTensor), 'building_mask': torch.from_numpy(mask).type(torch.LongTensor), 'damage_mask': torch.from_numpy(damage_class).type(torch.LongTensor), 'pre_image_orig': transforms.ToTensor()(pre_img_orig), 'post_image_orig': transforms.ToTensor()(post_img_orig), 'img_file_idx':imgs_dir[0:-1*(len(img_suffix))].split('/')[-1] + img_suffix, 'preds_img_dir':preds_dir}
