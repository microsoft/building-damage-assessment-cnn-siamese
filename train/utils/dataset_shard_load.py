# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import logging
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
import os

class DisasterDataset(Dataset):
    def __init__(self, data_dir, i_shard, set_name, data_mean_stddev, transform:bool, normalize:bool):
        
        self.data_dir = data_dir
        self.transform = transform
        self.normalize = normalize
        self.data_mean_stddev = data_mean_stddev
        
        shard_path = os.path.join(data_dir, f'{set_name}_pre_image_chips_{i_shard}.npy')
        self.pre_image_chip_shard = np.load(shard_path)
        logging.info(f'pre_image_chips loaded{self.pre_image_chip_shard.shape}')

        shard_path = os.path.join(data_dir, f'{set_name}_post_image_chips_{i_shard}.npy')
        self.post_image_chip_shard = np.load(shard_path)
        logging.info(f'post_image_chips loaded{self.post_image_chip_shard.shape}')

        shard_path = os.path.join(data_dir, f'{set_name}_bld_mask_chips_{i_shard}.npy')
        self.bld_mask_chip_shard = np.load(shard_path)
        logging.info(f'bld_mask_chips loaded{self.bld_mask_chip_shard.shape}')

        shard_path = os.path.join(data_dir, f'{set_name}_dmg_mask_chips_{i_shard}.npy')
        self.dmg_mask_chip_shard = np.load(shard_path)
        logging.info(f'dmg_mask_chips loaded{self.dmg_mask_chip_shard.shape}')

        shard_path = os.path.join(data_dir, f'{set_name}_pre_img_tile_chips_{i_shard}.npy')
        self.pre_img_tile_chip_shard = np.load(shard_path)
        logging.info(f'pre_img_tile_chips loaded{self.pre_img_tile_chip_shard.shape}')



    def __len__(self):
        return len(self.pre_image_chip_shard)
    
    @classmethod
    def apply_transform(self, mask, pre_img, post_img, damage_class):

        '''
        apply tranformation functions on cv2 arrays
        '''

        # Random horizontal flipping
        if random.random() > 0.5:
            mask = cv2.flip(mask, flipCode=1)
            pre_img = cv2.flip(pre_img, flipCode=1)
            post_img = cv2.flip(post_img, flipCode=1)
            damage_class = cv2.flip(damage_class, flipCode=1)

        # Random vertical flipping
        if random.random() > 0.5:
            mask = cv2.flip(mask, flipCode=0)
            pre_img = cv2.flip(pre_img, flipCode=0)
            post_img = cv2.flip(post_img, flipCode=0)
            damage_class = cv2.flip(damage_class, flipCode=0)

        return mask, pre_img, post_img, damage_class

    def __getitem__(self, i):
            
        pre_img = self.pre_image_chip_shard[i]
        post_img = self.post_image_chip_shard[i]
        mask = self.bld_mask_chip_shard[i]
        damage_class= self.dmg_mask_chip_shard[i]

        # copy original image for viz
        pre_img_orig = pre_img
        post_img_orig = post_img

        if self.transform is True:
            mask, pre_img, post_img, damage_class = self.apply_transform(mask, pre_img, post_img, damage_class)
        
        if self.normalize is True:
            pre_img_tile_name =   self.pre_img_tile_chip_shard[i]
            post_img_tile_name = pre_img_tile_name.replace('pre', 'post')
    
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
        else:
            pre_img = np.array(transforms.ToTensor()(pre_img)).astype(dtype='float64')/255.0
            post_img = np.array(transforms.ToTensor()(post_img)).astype(dtype='float64')/255.0

        # convert eveything to arrays
        pre_img = np.array(pre_img)
        post_img = np.array(post_img)
        mask = np.array(mask) 
        damage_class = np.array(damage_class)

        # replace non-classified pixels with background
        damage_class = np.where(damage_class==5, 0, damage_class)
        
        return {'pre_image': torch.from_numpy(pre_img).type(torch.FloatTensor), 'post_image': torch.from_numpy(post_img).type(torch.FloatTensor), 'building_mask': torch.from_numpy(mask).type(torch.LongTensor), 'damage_mask': torch.from_numpy(damage_class).type(torch.LongTensor), 'pre_image_orig': transforms.ToTensor()(pre_img_orig), 'post_image_orig': transforms.ToTensor()(post_img_orig)}
