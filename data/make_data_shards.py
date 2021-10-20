# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
make_data_shards.py

This is an additional pre-processing step after tile_and_mask.py to cut chips out of the tiles
and store them in large numpy arrays, so they can all be loaded in memory during training.

The train and val splits will be stored separately to distinguish them.

This is an improvement on the original approach of chipping during training using LandsatDataset, but it is an
extra step, so each new experiment requiring a different input size/set of channels would need to re-run
this step. Data augmentation is still added on-the-fly.

Example invocation:
```
export AZUREML_DATAREFERENCE_wcsorinoquia=/boto_disk_0/wcs_data/tiles/full_sr_median_2013_2014

python data/make_chip_shards.py --config_module_path training_wcs/experiments/elevation/elevation_2_config.py --out_dir /boto_disk_0/wcs_data/shards/full_sr_median_2013_2014_elevation
```
"""

import argparse
import importlib
import os
import sys
import math
import numpy as np
from dataset_shard_save import DisasterDataset
from train_utils import load_json_files, dump_json_files


from tqdm import tqdm

config = {'num_shards': 1,
          'out_dir': './xBD_sliced_augmented_20_alldisasters_final_mdl_npy/',
          'data_dir': './xBD_sliced_augmented_20_alldisasters/',
          'data_splits_json': './nlrc.building-damage-assessment/constants/splits/final_mdl_all_disaster_splits_sliced_img_augmented_20.json',
          'data_mean_stddev': './nlrc.building-damage-assessment/constants/splits/all_disaster_mean_stddev_tiles_0_1.json'}

def load_dataset():
    splits = load_json_files(config['data_splits_json'])
    data_mean_stddev = load_json_files(config['data_mean_stddev'])

    train_ls = [] 
    val_ls = []
    test_ls = []
    for item, val in splits.items():
        train_ls += val['train'] 
        val_ls += val['val']
        test_ls += val['test']

    xBD_train = DisasterDataset(config['data_dir'], train_ls, data_mean_stddev, transform=False, normalize=True)
    xBD_val = DisasterDataset(config['data_dir'], val_ls, data_mean_stddev, transform=False, normalize=True)
    xBD_test = DisasterDataset(config['data_dir'], test_ls, data_mean_stddev, transform=False, normalize=True)

    print('xBD_disaster_dataset train length: {}'.format(len(xBD_train)))
    print('xBD_disaster_dataset val length: {}'.format(len(xBD_val)))
    print('xBD_disaster_dataset test length: {}'.format(len(xBD_test)))

    return xBD_train, xBD_val, xBD_test

def create_shard(dataset, num_shards):
    """Iterate through the dataset to produce shards of chips as numpy arrays, for imagery input and labels.

    Args:
        dataset: an instance of LandsatDataset, which when iterated, each item contains fields
                    'chip' and 'chip_label'
        data = {'pre_image': pre_img, 'post_image': post_img, 'building_mask': mask, 'damage_mask': damage_class}

        num_shards: number of numpy arrays to store all chips in

    Returns:
        returns a 2-tuple, where
        - the first item is a list of numpy arrays of dimension (num_chips, channel, height, width) with
          dtype float for the input imagery chips
        - the second item is a list of numpy arrays of dimension (num_chips, height, width) with
          dtype int for the label chips.
    """
    pre_image_chips, post_image_chips, bld_mask_chips, dmg_mask_chips, pre_img_tile_name_chips = [], [], [], [], []
    for item in tqdm(dataset):
        # not using chip_id and chip_for_display fields
        pre_image_chips.append(item['pre_image'])
        post_image_chips.append(item['post_image'])
        bld_mask_chips.append(item['building_mask'])
        dmg_mask_chips.append(item['damage_mask'])
        pre_img_tile_name_chips.append(item['pre_img_tile_name'])

    num_chips = len(pre_image_chips)
    print(f'Created {num_chips} chips.')

    items_per_shards = math.ceil(num_chips / num_shards)
    shard_idx = []
    for i in range(num_shards):
        shard_idx.append(
            (i * items_per_shards, (1 + i) * items_per_shards)
        )

    print('Stacking imagery and label chips into shards')
    pre_image_chip_shards, post_image_chip_shards, bld_mask_chip_shards, dmg_mask_chip_shards, pre_img_tile_name_chip_shards = [], [], [], [], []
    for begin_idx, end_idx in shard_idx:
        if begin_idx < num_chips:
            pre_image_chip_shard = pre_image_chips[begin_idx:end_idx]
            pre_image_chip_shard = np.stack(pre_image_chip_shard, axis=0)
            print(f'dim of input chip shard is {pre_image_chip_shard.shape}, dtype is {pre_image_chip_shard.dtype}')
            pre_image_chip_shards.append(pre_image_chip_shard)

            post_image_chip_shard = post_image_chips[begin_idx:end_idx]
            post_image_chip_shard = np.stack(post_image_chip_shard, axis=0)
            print(f'dim of input chip shard is {post_image_chip_shard.shape}, dtype is {post_image_chip_shard.dtype}')
            post_image_chip_shards.append(post_image_chip_shard)

            bld_mask_chip_shard = bld_mask_chips[begin_idx:end_idx]
            bld_mask_chip_shard = np.stack(bld_mask_chip_shard, axis=0)
            print(f'dim of label chip shard is {bld_mask_chip_shard.shape}, dtype is {bld_mask_chip_shard.dtype}')
            bld_mask_chip_shards.append(bld_mask_chip_shard)

            dmg_mask_chip_shard = dmg_mask_chips[begin_idx:end_idx]
            dmg_mask_chip_shard = np.stack(dmg_mask_chip_shard, axis=0)
            print(f'dim of label chip shard is {dmg_mask_chip_shard.shape}, dtype is {dmg_mask_chip_shard.dtype}')
            dmg_mask_chip_shards.append(dmg_mask_chip_shard)

            pre_img_tile_name_chip_shard = pre_img_tile_name_chips[begin_idx:end_idx]
            pre_img_tile_name_chip_shard = np.stack(pre_img_tile_name_chip_shard, axis=0)
            print(f'dim of pre_img_tile_name_chip_shard chip shard is {pre_img_tile_name_chip_shard.shape}, dtype is {pre_img_tile_name_chip_shard.dtype}')
            pre_img_tile_name_chip_shards.append(pre_img_tile_name_chip_shard)

    return (pre_image_chip_shards, post_image_chip_shards, bld_mask_chip_shards, dmg_mask_chip_shards, pre_img_tile_name_chip_shards)


def save_shards(out_dir, set_name, pre_image_chip_shards, post_image_chip_shards, bld_mask_chip_shards, dmg_mask_chip_shards, pre_img_tile_name_chip_shards):
    for i_shard, (pre_image_chip_shard, post_image_chip_shard, bld_mask_chip_shard, dmg_mask_chip_shard, pre_img_tile_name_chip_shard) in enumerate(zip(pre_image_chip_shards, post_image_chip_shards, bld_mask_chip_shards, dmg_mask_chip_shards, pre_img_tile_name_chip_shards)):
        shard_path = os.path.join(out_dir, f'{set_name}_pre_image_chips_{i_shard}.npy')
        np.save(shard_path, pre_image_chip_shard)
        print(f'Saved {shard_path}')

        shard_path = os.path.join(out_dir, f'{set_name}_post_image_chips_{i_shard}.npy')
        np.save(shard_path, post_image_chip_shard)
        print(f'Saved {shard_path}')

        shard_path = os.path.join(out_dir, f'{set_name}_bld_mask_chips_{i_shard}.npy')
        np.save(shard_path, bld_mask_chip_shard)
        print(f'Saved {shard_path}')

        shard_path = os.path.join(out_dir, f'{set_name}_dmg_mask_chips_{i_shard}.npy')
        np.save(shard_path, dmg_mask_chip_shard)
        print(f'Saved {shard_path}')

        shard_path = os.path.join(out_dir, f'{set_name}_pre_img_tile_chips_{i_shard}.npy')
        np.save(shard_path, pre_img_tile_name_chip_shard)
        print(f'Saved {shard_path}')


def main():

    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    train_set, val_set, test_set = load_dataset()

    print('Iterating through the training set to generate chips...')
    train_pre_image_chip_shards, train_post_image_chip_shards, train_bld_mask_chip_shards, train_dmg_mask_chip_shards, train_pre_img_tile_name_chip_shards = create_shard(train_set, config['num_shards'])
    save_shards(out_dir, 'train', train_pre_image_chip_shards, train_post_image_chip_shards, train_bld_mask_chip_shards, train_dmg_mask_chip_shards, train_pre_img_tile_name_chip_shards)

    del train_pre_image_chip_shards
    del train_post_image_chip_shards
    del train_bld_mask_chip_shards
    del train_dmg_mask_chip_shards

    print('Iterating through the val set to generate chips...')
    val_pre_image_chip_shards, val_post_image_chip_shards, val_bld_mask_chip_shards, val_dmg_mask_chip_shards, val_pre_img_tile_name_chip_shards = create_shard(val_set, config['num_shards'])
    save_shards(out_dir, 'val', val_pre_image_chip_shards, val_post_image_chip_shards, val_bld_mask_chip_shards, val_dmg_mask_chip_shards, val_pre_img_tile_name_chip_shards)

    del val_pre_image_chip_shards
    del val_post_image_chip_shards
    del val_bld_mask_chip_shards
    del val_dmg_mask_chip_shards

    print('Done!')


if __name__ == '__main__':
    main()