# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import json
import torch
import argparse
import logging
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from datetime import datetime
from torchvision import transforms
from data.raster_label_visualizer import RasterLabelVisualizer
from models.end_to_end_Siam_UNet import SiamUnet
from utils.datasets import DisasterDataset
from eval.eval_building_level import _evaluate_tile, get_label_and_pred_polygons_for_tile_mask_input, allowed_classes
from torch.utils.tensorboard import SummaryWriter


config = {'labels_dmg': [0, 1, 2, 3, 4],
          'labels_bld': [0, 1]}
          
parser = argparse.ArgumentParser(description='Building Damage Assessment Inference')
parser.add_argument('--output_dir', type=str, required=True, help='Path to an empty directory where outputs will be saved. This directory will be created if it does not exist.')
parser.add_argument('--data_img_dir', type=str, required=True, help='Path to a directory that contains input images.')
parser.add_argument('--data_inference_dict', type=str, required=True, help='Path to a json file that contains a dict of path information for each individual image to be used for inference.')
parser.add_argument('--data_mean_stddev', type=str, required=True, help='Path to a json file that contains mean and stddev for each tile used for normalization of each image patch.')
parser.add_argument('--label_map_json', type=str, required=True, help='Path to a json file that contains information between actual labels and encoded labels for classification task.')
parser.add_argument('--model', type=str, required=True, help='Path to a trained model to be used for inference.')
parser.add_argument('--gpu', type=str, default="cuda:0", help='GPU to run on.')
parser.add_argument('--experiment_name', type=str, default='infer', help='Choose a name for each inference test folder.')
parser.add_argument('--num_chips_to_viz', type=int, default=1, help='Number of patches to visualize in tensorboard for monitoring.')
args = parser.parse_args()

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='{asctime} {levelname} {message}',
                    style='{',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.info(f'Using PyTorch version {torch.__version__}.')
device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
logging.info(f'Using device: {device}.')


def main():

    eval_results_val_dmg = pd.DataFrame(columns=['class', 'precision', 'recall', 'f1', 'accuracy'])
    eval_results_val_dmg_building_level = pd.DataFrame(columns=['class', 'precision', 'recall', 'f1', 'accuracy'])
    eval_results_val_bld = pd.DataFrame(columns=['class', 'precision', 'recall', 'f1', 'accuracy'])

    # set up directories    
    logger_dir = os.path.join(args.output_dir, args.experiment_name, 'logs')
    os.makedirs(logger_dir, exist_ok=True)
    
    evals_dir = os.path.join(args.output_dir, args.experiment_name, 'evals')
    os.makedirs(evals_dir, exist_ok=True)

    output_dir = os.path.join(args.output_dir, args.experiment_name, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # initialize logger instances
    logger_test= SummaryWriter(log_dir=logger_dir)

    # load test data
    global test_dataset, test_loader, labels_set_dmg, labels_set_bld, viz
    label_map = load_json_files(args.label_map_json)
    viz = RasterLabelVisualizer(label_map=label_map)
    test_dataset, samples_idx_list = load_dataset()
    
    labels_set_dmg = config['labels_dmg']
    labels_set_bld = config['labels_bld']

    #load model and its state from the given checkpoint
    model = SiamUnet()
    checkpoint_path = args.model
    if checkpoint_path and os.path.isfile(checkpoint_path):
        logging.info('Loading checkpoint from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device=device)
        logging.info(f"Using checkpoint at epoch {checkpoint['epoch']}, val f1 is {checkpoint.get('val_f1_avg', 'Not Available')}")
    else:
        logging.info('No valid checkpoint is provided.')
        return
    
    # inference
    logging.info(f'Start model inference ...')
    inference_start_time = datetime.now()
    confusion_mtrx_df_val_dmg, confusion_mtrx_df_val_bld, confusion_mtrx_df_val_dmg_building_level = validate(test_dataset, model, logger_test, samples_idx_list, evals_dir)                
    inference_duration = datetime.now() - inference_start_time
    logging.info((f'inference duration is {inference_duration.total_seconds()} seconds'))

    logging.info(f'compute actual metrics for model evaluation based on validation set ...')

    # damage level eval validation (pixelwise)
    eval_results_val_dmg = compute_eval_metrics(labels_set_dmg, confusion_mtrx_df_val_dmg, eval_results_val_dmg)
    f1_harmonic_mean = 0
    metrics = 'f1'
    for index, row in eval_results_val_dmg.iterrows():
        if (int(row['class']) in labels_set_dmg[1:]) & (metrics == 'f1'):
            f1_harmonic_mean += 1.0/(row[metrics]+1e-10)
    f1_harmonic_mean = 4.0/f1_harmonic_mean
    eval_results_val_dmg = eval_results_val_dmg.append({'class':'harmonic-mean-of-all', 'precision':'-', 'recall':'-', 'f1':f1_harmonic_mean, 'accuracy':'-'}, ignore_index=True)
    
    # damage level eval validation (building-level)
    eval_results_val_dmg_building_level = compute_eval_metrics(labels_set_dmg, confusion_mtrx_df_val_dmg_building_level, eval_results_val_dmg_building_level)
    f1_harmonic_mean = 0
    metrics = 'f1'
    for index, row in eval_results_val_dmg_building_level.iterrows():
        if (int(row['class']) in labels_set_dmg[1:]) & (metrics == 'f1'):
            f1_harmonic_mean += 1.0/(row[metrics]+1e-10)
    f1_harmonic_mean = 4.0/f1_harmonic_mean
    eval_results_val_dmg_building_level = eval_results_val_dmg_building_level.append({'class':'harmonic-mean-of-all', 'precision':'-', 'recall':'-', 'f1':f1_harmonic_mean, 'accuracy':'-'}, ignore_index=True)
    

    # bld detection eval validation (pixelwise)
    eval_results_val_bld = compute_eval_metrics(labels_set_bld, confusion_mtrx_df_val_bld, eval_results_val_bld)

    # save confusion metrices
    confusion_mtrx_df_val_bld.to_csv(os.path.join(evals_dir, 'confusion_mtrx_bld.csv'), index=False)
    confusion_mtrx_df_val_dmg.to_csv(os.path.join(evals_dir, 'confusion_mtrx_dmg.csv'), index=False)
    confusion_mtrx_df_val_dmg_building_level.to_csv(os.path.join(evals_dir, 'confusion_mtrx_dmg_building_level.csv'), index=False)
    
    # save evalution metrics
    eval_results_val_bld.to_csv(os.path.join(evals_dir, 'eval_results_bld.csv'), index=False)
    eval_results_val_dmg.to_csv(os.path.join(evals_dir, 'eval_results_dmg.csv'), index=False)
    eval_results_val_dmg_building_level.to_csv(os.path.join(evals_dir, 'eval_results_dmg_building_level.csv'), index=False)

    logging.info('Done')

    return

def validate(loader, model, logger_test, samples_idx_list, evals_dir):
    
    """
    Evaluate the model on dataset of the loader
    """
    softmax = torch.nn.Softmax(dim=1)
    model.eval()  # put model to evaluation mode
    confusion_mtrx_df_val_dmg = pd.DataFrame(columns=['img_idx', 'class', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'total'])
    confusion_mtrx_df_val_bld = pd.DataFrame(columns=['img_idx', 'class', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'total'])
    confusion_mtrx_df_val_dmg_building_level = pd.DataFrame(columns=['img_idx', 'class', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'total'])

    with torch.no_grad():
        for img_idx, data in enumerate(tqdm(loader)): # assume batch size is 1
            c = data['pre_image'].size()[0]
            h = data['pre_image'].size()[1]
            w = data['pre_image'].size()[2]

            x_pre = data['pre_image'].reshape(1, c, h, w).to(device=device)
            x_post = data['post_image'].reshape(1, c, h, w).to(device=device)
            y_seg = data['building_mask'].to(device=device)  
            y_cls = data['damage_mask'].to(device=device)

            scores = model(x_pre, x_post)
                    
            # compute accuracy for segmenation model on pre_ images
            preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)

            # modify damage prediction based on UNet arm predictions       
            for c in range(0,scores[2].shape[1]):
                scores[2][:,c,:,:] = torch.mul(scores[2][:,c,:,:], preds_seg_pre)
            preds_cls = torch.argmax(softmax(scores[2]), dim=1)
            
            path_pred_mask = data['preds_img_dir'] +'.png'
            logging.info('save png image for damage level predictions: ' + path_pred_mask)
            im = Image.fromarray(preds_cls.cpu().numpy()[0,:,:].astype(np.uint8))
            if not os.path.exists(os.path.split(data['preds_img_dir'])[0]):
                os.makedirs(os.path.split(data['preds_img_dir'])[0])
            im.save(path_pred_mask)
            logging.info(f'saved image size: {preds_cls.size()}')

            # compute building-level confusion metrics
            pred_polygons_and_class, label_polygons_and_class = get_label_and_pred_polygons_for_tile_mask_input(y_cls.cpu().numpy().astype(np.uint8), path_pred_mask)
            results, list_preds, list_labels = _evaluate_tile(pred_polygons_and_class, label_polygons_and_class, allowed_classes, 0.1)
            total_objects = results[-1]
            for label_class in results:
                if label_class != -1:
                    true_pos_cls = results[label_class]['tp'] if 'tp' in results[label_class].keys() else 0
                    true_neg_cls = results[label_class]['tn'] if 'tn' in results[label_class].keys() else 0
                    false_pos_cls = results[label_class]['fp'] if 'fp' in results[label_class].keys() else 0
                    false_neg_cls = results[label_class]['fn'] if 'fn' in results[label_class].keys() else 0
                    confusion_mtrx_df_val_dmg_building_level = confusion_mtrx_df_val_dmg_building_level.append({'img_idx':img_idx, 'class':label_class, 'true_pos':true_pos_cls, 'true_neg':true_neg_cls, 'false_pos':false_pos_cls, 'false_neg':false_neg_cls, 'total':total_objects}, ignore_index=True)

            # compute comprehensive pixel-level comfusion metrics
            confusion_mtrx_df_val_dmg = compute_confusion_mtrx(confusion_mtrx_df_val_dmg, img_idx, labels_set_dmg, preds_cls, y_cls, y_seg)
            confusion_mtrx_df_val_bld = compute_confusion_mtrx(confusion_mtrx_df_val_bld, img_idx, labels_set_bld, preds_seg_pre, y_seg, [])

            # add viz results to logger
            if img_idx in samples_idx_list:
                prepare_for_vis(img_idx, logger_test, model, device, softmax)

    return confusion_mtrx_df_val_dmg, confusion_mtrx_df_val_bld, confusion_mtrx_df_val_dmg_building_level

def load_dataset():
    splits = load_json_files(args.data_inference_dict)
    data_mean_stddev = load_json_files(args.data_mean_stddev)    
    test_ls = []
    for item, val in splits.items():
        test_ls += val['test']
    test_dataset = DisasterDataset(args.data_img_dir, test_ls, data_mean_stddev, transform=False, normalize=True)
    logging.info('xBD_disaster_dataset test length: {}'.format(len(test_dataset)))
    assert len(test_dataset) > 0
    samples_idx_list = get_sample_images(test_dataset)
    logging.info('items selected for viz: {}'.format(samples_idx_list))
    return test_dataset, samples_idx_list

def compute_eval_metrics(labels_set, confusion_mtrx_df, eval_results):
    for cls in labels_set: 
        class_idx = (confusion_mtrx_df['class']==cls)
        precision = confusion_mtrx_df.loc[class_idx,'true_pos'].sum()/(confusion_mtrx_df.loc[class_idx,'true_pos'].sum() + confusion_mtrx_df.loc[class_idx,'false_pos'].sum() + sys.float_info.epsilon)
        recall = confusion_mtrx_df.loc[class_idx,'true_pos'].sum()/(confusion_mtrx_df.loc[class_idx,'true_pos'].sum() + confusion_mtrx_df.loc[class_idx,'false_neg'].sum() + sys.float_info.epsilon)
        f1 = 2 * (precision * recall)/(precision + recall + sys.float_info.epsilon)
        accuracy = (confusion_mtrx_df.loc[class_idx,'true_pos'].sum() + confusion_mtrx_df.loc[class_idx,'true_neg'].sum())/(confusion_mtrx_df.loc[class_idx,'total'].sum() + sys.float_info.epsilon)
        eval_results = eval_results.append({'class':cls, 'precision':precision, 'recall':recall, 'f1':f1, 'accuracy':accuracy}, ignore_index=True)
    return eval_results

def compute_confusion_mtrx(confusion_mtrx_df, img_idx, labels_set, y_preds, y_true, y_true_bld_mask):
    for cls in labels_set[1:]:
        confusion_mtrx_df = compute_confusion_mtrx_class(confusion_mtrx_df, img_idx, labels_set, y_preds, y_true, y_true_bld_mask, cls)
    return confusion_mtrx_df

def compute_confusion_mtrx_class(confusion_mtrx_df, img_idx, labels_set, y_preds, y_true, y_true_bld_mask, cls):

    y_true_binary = y_true.detach().clone()
    y_preds_binary = y_preds.detach().clone()
    
    if len(labels_set) > 2:
        # convert to 0/1
        y_true_binary[y_true_binary != cls] = -1
        y_preds_binary[y_preds_binary != cls] = -1
        
        y_true_binary[y_true_binary == cls] = 1
        y_preds_binary[y_preds_binary == cls] = 1
        
        y_true_binary[y_true_binary == -1] = 0
        y_preds_binary[y_preds_binary == -1] = 0
        
        # compute confusion metric
        true_pos_cls = ((y_true_binary == y_preds_binary) & (y_true_binary == 1) & (y_true_bld_mask == 1)).float().sum().item()
        false_neg_cls = ((y_true_binary != y_preds_binary) & (y_true_binary == 1) & (y_true_bld_mask == 1)).float().sum().item()
        true_neg_cls = ((y_true_binary == y_preds_binary) & (y_true_binary == 0) & (y_true_bld_mask == 1)).float().sum().item()
        false_pos_cls = ((y_true_binary != y_preds_binary) & (y_true_binary == 0) & (y_true_bld_mask == 1)).float().sum().item()
        
        # compute total pixels
        total_pixels = y_true_bld_mask.float().sum().item()

    else:

        # compute confusion metric
        true_pos_cls = ((y_true_binary == y_preds_binary) & (y_true_binary == 1)).float().sum().item()
        false_neg_cls = ((y_true_binary != y_preds_binary) & (y_true_binary == 1)).float().sum().item()
        true_neg_cls = ((y_true_binary == y_preds_binary) & (y_true_binary == 0)).float().sum().item()
        false_pos_cls = ((y_true_binary != y_preds_binary) & (y_true_binary == 0)).float().sum().item()
        
        # compute total pixels
        total_pixels = 1
        for item in y_true_binary.size():
            total_pixels *= item

    confusion_mtrx_df = confusion_mtrx_df.append({'class':cls, 'img_idx':img_idx, 'true_pos':true_pos_cls, 'true_neg':true_neg_cls, 'false_pos':false_pos_cls, 'false_neg':false_neg_cls, 'total':total_pixels}, ignore_index=True)
    return confusion_mtrx_df

def prepare_for_vis(item, logger, model, device, softmax):
    
    iteration = 0
    data = test_dataset[item]
    c = data['pre_image'].size()[0]
    h = data['pre_image'].size()[1]
    w = data['pre_image'].size()[2]

    pre = data['pre_image'].reshape(1, c, h, w)
    post = data['post_image'].reshape(1, c, h, w)
    
    scores = model(pre.to(device=device), post.to(device=device))
    preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
    preds_seg_post = torch.argmax(softmax(scores[1]), dim=1)
    
    # modify damage prediction based on UNet arm predictions       
    for c in range(0,scores[2].shape[1]):
        scores[2][:,c,:,:] = torch.mul(scores[2][:,c,:,:], preds_seg_pre)

    # visualize predictions and add to tensorboard
    tag = 'pr_bld_mask_pre_test_id_' + str(item)
    logger.add_image(tag, preds_seg_pre, iteration, dataformats='CHW')
    
    tag = 'pr_bld_mask_post_test_id_' + str(item)
    logger.add_image(tag, preds_seg_post, iteration, dataformats='CHW')
    
    tag = 'pr_dmg_mask_test_id_' + str(item)
    im, buf = viz.show_label_raster(torch.argmax(softmax(scores[2]), dim=1).cpu().numpy(), size=(5, 5))
    preds_cls = transforms.ToTensor()(transforms.ToPILImage()(np.array(im)).convert("RGB"))
    logger.add_image(tag, preds_cls, iteration, dataformats='CHW')
                
    # visualize GT and add to tensorboard
    pre = data['pre_image']
    tag = 'gt_img_pre_test_id_' + str(item)
    logger.add_image(tag, data['pre_image_orig'], iteration, dataformats='CHW')
    
    post = data['post_image']
    tag = 'gt_img_post_test_id_' + str(item)
    logger.add_image(tag, data['post_image_orig'], iteration, dataformats='CHW')

    gt_seg = data['building_mask'].reshape(1, h, w)
    tag = 'gt_bld_mask_test_id_' + str(item)
    logger.add_image(tag, gt_seg, iteration, dataformats='CHW')

    im, buf = viz.show_label_raster(np.array(data['damage_mask']), size=(5, 5))
    gt_cls = transforms.ToTensor()(transforms.ToPILImage()(np.array(im)).convert("RGB"))
    tag = 'gt_dmg_mask_test_id_' + str(item)
    logger.add_image(tag, gt_cls, iteration, dataformats='CHW')    
    return

def get_sample_images(dataset):

    assert len(dataset) > args.num_chips_to_viz

    samples_idx_list = []
    from random import randint
    for sample_idx in range(0, args.num_chips_to_viz):
        value = randint(0, len(dataset))
        samples_idx_list.append(value)
    
    return samples_idx_list

def load_json_files(json_filename):
    with open(json_filename) as f:
        file_content = json.load(f)
    return file_content

def dump_json_files(json_filename, my_dict):
    with open(json_filename, 'w') as f:
        json.dump(my_dict, f, indent=4) 
    return   

if __name__ == "__main__":
    main()