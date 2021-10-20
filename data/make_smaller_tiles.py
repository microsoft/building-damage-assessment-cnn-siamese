# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import copy
import json
import logging
import os
import random
import sys
from glob import glob
from multiprocessing.pool import ThreadPool
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

config = {
    'batch_size': 1,
    'data_dir': './xBD/',
    'sliced_data_dir': './final_mdl_all_disaster_splits/',
    'disaster_splits_json': './nlrc.building-damage-assessment/constants/splits/final_mdl_all_disaster_splits.json',
    'disaster_splits_json_sliced': './nlrc.building-damage-assessment/constants/splits/final_mdl_all_disaster_splits_sliced_img_augmented_20.json'
}

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='{asctime} {levelname} {message}',
    style='{',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def main():

    xBD_train, xBD_val, xBD_test = load_dataset()

    train_loader = DataLoader(xBD_train, batch_size=config['batch_size'], shuffle=False, num_workers=8)
    test_loader = DataLoader(xBD_test, batch_size=config['batch_size'], shuffle=False, num_workers=8)
    val_loader = DataLoader(xBD_val, batch_size=config['batch_size'], shuffle=False, num_workers=8)

    with ThreadPool(3) as pool:
        loaders = [
            ('TRAIN', train_loader),
            ('TEST', test_loader),
            ('VAL', val_loader)
        ]
        pool.starmap(iterate_and_slice, loaders)

    logging.info(f'Done')
    
    return


def iterate_and_slice(split_name: str, data_loader: DataLoader):
    for batch_idx, data in enumerate(data_loader):
        logging.info(f'{split_name}: batch_idx {batch_idx} sliced into 20 images.')
    logging.info(f'Done for split {split_name}')


def load_dataset():
    splits_all_disasters = load_json_files(config['disaster_splits_json'])
    sliced_data_json = copy.deepcopy(splits_all_disasters)

    train_ls = [] 
    val_ls = []
    test_ls = []
    for disaster_name, splits in splits_all_disasters.items():
        logging.info(f'disaster_name: {disaster_name}.')
        l = len(splits['train'])
        logging.info(f'training set number of tiles: {l}.')

        train_ls += splits['train']
        val_ls += splits['val']
        test_ls += splits['test']
 
    for disaster_name,  splits in sliced_data_json.items():
        new_vals_tr = []
        new_vals_ts = []
        new_vals_val = []

        for slice_sub in range(0,20):
            new_vals_tr += [sub + '_' + str(slice_sub) for sub in sliced_data_json[disaster_name]['train']]
            new_vals_ts += [sub + '_' + str(slice_sub) for sub in sliced_data_json[disaster_name]['test']]
            new_vals_val += [sub + '_' + str(slice_sub) for sub in sliced_data_json[disaster_name]['val']]

        logging.info(f'disaster_name: {disaster_name}.')
        logging.info(f'training set number of chips: {len(new_vals_tr)}.')

        sliced_data_json[disaster_name]['train'] = new_vals_tr
        sliced_data_json[disaster_name]['test'] = new_vals_ts
        sliced_data_json[disaster_name]['val'] = new_vals_val
    
    dump_json_files(config['disaster_splits_json_sliced'], sliced_data_json)

    logging.info(f'train dataset length before cropping: {len(train_ls)}.')
    xBD_train = SliceDataset(config['data_dir'], train_ls, config['sliced_data_dir'])
    logging.info(f'train dataset length after cropping: {len(xBD_train)}.')

    logging.info(f'test dataset length before cropping: {len(test_ls)}.')
    xBD_test = SliceDataset(config['data_dir'], test_ls, config['sliced_data_dir'])
    logging.info(f'test dataset length after cropping: {len(xBD_test)}.')

    logging.info(f'val dataset length before cropping: {len(val_ls)}.')
    xBD_val = SliceDataset(config['data_dir'], val_ls, config['sliced_data_dir'])
    logging.info(f'val dataset length after cropping: {len(xBD_val)}.')

    print('xBD_disaster_dataset train length: {}'.format(len(xBD_train)))
    print('xBD_disaster_dataset val length: {}'.format(len(xBD_val)))
    print('xBD_disaster_dataset test length: {}'.format(len(xBD_test)))
    return xBD_train, xBD_val, xBD_test


def load_json_files(json_filename):
    with open(json_filename) as f:
        file_content = json.load(f)
    return file_content


def dump_json_files(json_filename, my_dict):
    with open(json_filename, 'w') as f:
        json.dump(my_dict, f, indent=4) 
    return


class SliceDataset(Dataset):
    def __init__(self, data_dir, data_dir_ls, sliced_data_dir, mask_suffix=''):
        """
        Args:
            data_dir: root xBD directory
            data_dir_ls: list of all tiles in this split, across different disasters
            sliced_data_dir: output directory for chips
        """
        
        self.data_dir = data_dir
        self.data_dir_ls = data_dir_ls
        self.sliced_data_dir = sliced_data_dir
    
    def __len__(self):
        return len(self.data_dir_ls)

    @classmethod
    def slice_tile(self, mask, pre_img, post_img, damage_class):
        

        img_h = pre_img.size[0]
        img_w = pre_img.size[1]
        
        h_idx = [0, 256, 512, 768]
        w_idx = [0, 256, 512, 768]

        img_h = 256
        img_w = 256
        
        sliced_sample_dic = {}
        counter = 0
        for i in h_idx: 
            for j in w_idx: 
                mask_sub = TF.crop(mask, i, j, img_h, img_w)
                pre_img_sub = TF.crop(pre_img, i, j, img_h, img_w)
                post_img_sub = TF.crop(post_img, i, j, img_h, img_w)
                damage_class_sub = TF.crop(damage_class, i, j, img_h, img_w)
                sliced_sample_dic[str(counter)] = {'mask': mask_sub, 'pre_img': pre_img_sub, 'post_img': post_img_sub, 'damage_class': damage_class_sub}
                counter += 1
        
        # pick 4 random slices from each tile
        for item in range(0,4):
            i = random.randint(5, h_idx[-1]-5)
            j = random.randint(5, w_idx[-1]-5)
            mask_sub = TF.crop(mask, i, j, img_h, img_w)
            pre_img_sub = TF.crop(pre_img, i, j, img_h, img_w)
            post_img_sub = TF.crop(post_img, i, j, img_h, img_w)
            damage_class_sub = TF.crop(damage_class, i, j, img_h, img_w)
            sliced_sample_dic[str(counter)] = {'mask': mask_sub, 'pre_img': pre_img_sub, 'post_img': post_img_sub, 'damage_class': damage_class_sub}
            counter += 1

        return sliced_sample_dic

    def __getitem__(self, i):
        
        imgs_dir = os.path.join(self.data_dir, self.data_dir_ls[i].replace('labels', 'images'))
        masks_dir = os.path.join(self.data_dir, self.data_dir_ls[i].replace('labels', 'targets_border2'))

        idx = imgs_dir
        
        pre_img_file_name = imgs_dir + '_pre_disaster'
        pre_img_file = glob(pre_img_file_name + '.*')
        
        mask_file_name = masks_dir + '_pre_disaster_b2' 
        mask_file = glob(mask_file_name + '.*')

        post_img_file_name = pre_img_file_name.replace('pre', 'post')
        post_img_file = glob(post_img_file_name + '.*')

        damage_class_file_name = mask_file_name.replace('pre', 'post')
        damage_class_file = glob(damage_class_file_name + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(pre_img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {pre_img_file}'
        assert len(post_img_file) == 1, \
            f'Either no post disaster image or multiple images found for the ID {idx}: {post_img_file}'
        assert len(damage_class_file) == 1, \
            f'Either no damage class image or multiple images found for the ID {idx}: {damage_class_file}'

        mask = Image.open(mask_file[0])
        pre_img = Image.open(pre_img_file[0])
        post_img = Image.open(post_img_file[0])
        damage_class = Image.open(damage_class_file[0])

        assert pre_img.size == mask.size, \
            f'Image and building mask {idx} should be the same size, but are {pre_img.size} and {mask.size}'
        assert pre_img.size == damage_class.size, \
            f'Image and damage classes mask {idx} should be the same size, but are {pre_img.size} and {damage_class.size}'
        assert pre_img.size == post_img.size, \
            f'Pre & post disaster Images {idx} should be the same size, but are {pre_img.size} and {post_img.size}'

        sliced_sample_dic = self.slice_tile(mask, pre_img, post_img, damage_class)

        for item, val in sliced_sample_dic.items():
            os.makedirs(
                os.path.split(mask_file[0].replace(self.data_dir, self.sliced_data_dir))[0],
                exist_ok=True
            )
            val['mask'].save(mask_file[0].replace(self.data_dir, self.sliced_data_dir).replace('.png', '_' + item + '.png'))
            file_name = mask_file[0].replace(self.data_dir, self.sliced_data_dir).replace('.png', '_' + item + '.png')

            os.makedirs(
                os.path.split(pre_img_file[0].replace(self.data_dir, self.sliced_data_dir))[0],
                exist_ok=True
            )
            val['pre_img'].save(pre_img_file[0].replace(self.data_dir, self.sliced_data_dir).replace('.png', '_' + item + '.png'))

            os.makedirs(
                os.path.split(post_img_file[0].replace(self.data_dir, self.sliced_data_dir))[0],
                exist_ok=True
            )
            val['post_img'].save(post_img_file[0].replace(self.data_dir, self.sliced_data_dir).replace('.png', '_' + item + '.png'))

            os.makedirs(
                os.path.split(damage_class_file[0].replace(self.data_dir, self.sliced_data_dir))[0],
                exist_ok=True
            )
            val['damage_class'].save(damage_class_file[0].replace(self.data_dir, self.sliced_data_dir).replace('.png', '_' + item + '.png'))
            
        return {
            'pre_image': TF.to_tensor(pre_img),
            'post_image': TF.to_tensor(post_img),
            'building_mask': TF.to_tensor(mask),
            'damage_mask': TF.to_tensor(damage_class)
        }


if __name__ == "__main__":
    main()
