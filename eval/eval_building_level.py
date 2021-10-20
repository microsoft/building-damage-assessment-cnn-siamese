# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""

Input a set of raster model output files (same format as the masks) and original polygon JSON files, produces
polygonized versions of the model output and compute metrics based on an IoU threshold.

TODO: we need the outer loop to iterate over tiles, and to draw confusion matrices etc.

"""

import json
import os
from typing import List, Tuple
from collections.abc import Iterable
from collections import defaultdict
import cv2
import rasterio.features
import numpy as np
import shapely.geometry
from shapely.geometry import mapping, Polygon
from PIL import Image

from data.create_label_masks import get_feature_info, read_json

all_classes = set([1, 2, 3, 4, 5])
allowed_classes = set([1, 2, 3, 4])  # 5 is Unclassified, not used during evaluation


def _evaluate_tile(pred_polygons_and_class: list, 
                  label_polygons_and_class: list,
                  allowed_classes,
                  iou_threshold: float=0.5):
    """
    Method
    - For each predicted polygon, we find the maximum value of IoU it has with any ground truth
    polygon within the tile. This ground truth polygon is its "match".
    - Using the threshold IoU specified (typically and by default 0.5), if a prediction has
    overlap above the threshold AND the correct class, it is considered a true positive.
    All other predictions, no matter what their IOU is with any gt, are false positives.
           - Note that it is possible for one ground truth polygon to be the match for
           multiple predictions, especially if the IoU threshold is low, but each prediction
           only has one matching ground truth polygon.
    - For ground truth polygon not matched by any predictions, it is a false negative.
    - Given the TP, FP, and FN counts for each class, we can calculate the precision and recall
    for each tile *for each class*.


    - To plot a confusion table, we output two lists, one for the predictions and one for the
    ground truth polygons (because the set of polygons to confuse over are not the same...)
    1. For the list of predictions, each item is associated with the ground truth class of
    the polygon that it matched, or a "false positive" attribute.
    2. For the list of ground truth polygons, each is associated with the predicted class of
    the polygon it matched, or a "false negative" attribute.

    Args:
        pred_polygons_and_class: list of tuples of shapely Polygon representing the geometry of the prediction,
            and the predicted class
        label_polygons_and_class: list of tuples of shapely Polygon representing the ground truth geometry,
            and the class
        allowed_classes: which classes should be evaluated
        iou_threshold: Intersection over union threshold above which a predicted polygon is considered
            true positive

    Returns:
        results: a dict of dicts, keyed by the class number, and points to a dict with counts of
            true positives "tp", false positives "fp", and false negatives "fn"
        list_preds: a list with one entry for each prediction. Each entry is of the form
            {'pred': 3, 'label': 3}. This information is for a confusion matrix based on the
            predicted polygons.
        list_labels: same as list_preds, while each entry corresponds to a ground truth polygon.
            The value for 'pred' is None if this polygon is a false negative.
    """

    # the matched label polygon's IoU with the pred polygon, and the label polygon's index
    pred_max_iou_w_label = [(0.0, None)] * len(pred_polygons_and_class)

    for i_pred, (pred_poly, pred_class) in enumerate(pred_polygons_and_class):

        # cannot skip pred_class if it's not in the allowed list, as the list above relies on their indices

        for i_label, (label_poly, label_class) in enumerate(label_polygons_and_class):

            if not pred_poly.is_valid:
                pred_poly = pred_poly.buffer(0)
            if not label_poly.is_valid:
                label_poly = label_poly.buffer(0)

            intersection = pred_poly.intersection(label_poly)
            union = pred_poly.union(label_poly)  # they should not have zero area
            iou = intersection.area / union.area

            if iou > pred_max_iou_w_label[i_pred][0]:
                pred_max_iou_w_label[i_pred] = (iou, i_label)

    results = defaultdict(lambda: defaultdict(int))  # class: {tp, fp, fn} counts
    results[-1] = len(pred_polygons_and_class)
    i_label_polygons_matched = set()
    list_preds = []
    list_labels = []

    for i_pred, (pred_poly, pred_class) in enumerate(pred_polygons_and_class):

        if pred_class not in allowed_classes:
            continue

        max_iou, matched_i_label = pred_max_iou_w_label[i_pred]

        item = {
            'pred': pred_class,
            'label': label_polygons_and_class[matched_i_label][1] if matched_i_label is not None else None
        }

        if matched_i_label is not None:
            list_labels.append(item)
            
        list_preds.append(item)
        

        if max_iou > iou_threshold and label_polygons_and_class[matched_i_label][1] == pred_class:
            # true positive
            i_label_polygons_matched.add(matched_i_label)
            results[pred_class]['tp'] += 1
            for cls in allowed_classes:
                if cls != pred_class:
                     results[cls]['tn'] += 1
        else:
            # false positive - all other predictions
            results[pred_class]['fp'] += 1  # note that it is a FP for the prediction's class
            # print(matched_i_label)
            ##results[matched_i_label]['fn'] += 1  # note that it is a FP for the prediction's class

    # calculate the number of false negatives - how many label polygons are not matched by any predictions
    for i_label, (label_poly, label_class) in enumerate(label_polygons_and_class):

        if label_class not in allowed_classes:
            continue

        if i_label not in i_label_polygons_matched:
            results[label_class]['fn'] += 1
            list_labels.append({
                'pred': None,
                'label': label_class
            })

    return results, list_preds, list_labels


def get_label_and_pred_polygons_for_tile_json_input(path_label_json, path_pred_mask):
    """
    For each tile, cast the polygons specified in the label JSON file to shapely Polygons, and
    polygonize the prediction mask.

    Args:
        path_label_json: path to the label JSON file provided by xBD
        path_pred_mask: path to the PNG or TIFF mask predicted by the model, where each pixel is one
            of the allowed classes.

    Returns:
        pred_polygons_and_class: list of tuples of shapely Polygon representing the geometry of the prediction,
            and the predicted class
        label_polygons_and_class: list of tuples of shapely Polygon representing the ground truth geometry,
            and the class
    """
    assert path_label_json.endswith('.json')

    # get the label polygons

    label_json = read_json(path_label_json)
    polys = get_feature_info(label_json)

    label_polygons_and_class = []  # tuples of (shapely polygon, damage_class_num)

    for uid, tup in polys.items():
        poly, damage_class_num = tup  # poly is a np.ndarray
        polygon = Polygon(poly)

        if damage_class_num in allowed_classes:
            label_polygons_and_class.append((polygon, damage_class_num))

    # polygonize the prediction mask

    # 1. Detect the connected components by all non-background classes to determine the predicted
    # building blobs first (if we do this per class, a building with some pixels predicted to be
    # in another class will result in more buildings than connected components)
    mask_pred = np.asarray(Image.open(path_pred_mask))
    assert len(mask_pred.shape) == 2, 'mask should be 2D only.'

    background_and_others_mask = np.where(mask_pred > 0, 1, 0).astype(np.int16)  # all non-background classes become 1

    # rasterio.features.shapes:
    # default is 4-connected for connectivity - see https://www.mathworks.com/help/images/pixel-connectivity.html
    # specify the `mask` parameter, otherwise the background will be returned as a shape
    connected_components = rasterio.features.shapes(background_and_others_mask, mask=mask_pred > 0)

    polygons = []
    for component_geojson, pixel_val in connected_components:
        # reference: https://shapely.readthedocs.io/en/stable/manual.html#python-geo-interface
        shape = shapely.geometry.shape(component_geojson)
        assert isinstance(shape, Polygon)
        if shape.area >20:
            polygons.append(shape)

    # 2. The majority class for each building blob is assigned to be that building's predicted class.
    polygons_by_class = []

    for c in all_classes:

        # default is 4-connected for connectivity
        shapes = rasterio.features.shapes(mask_pred, mask=mask_pred == c)

        for shape_geojson, pixel_val in shapes:
            shape = shapely.geometry.shape(shape_geojson)
            assert isinstance(shape, Polygon)
            polygons_by_class.append((shape, int(pixel_val)))

    # we take the class of the shape with the maximum overlap with the building polygon to be the class of the building - majority vote
    polygons_max_overlap = [0.0] * len(polygons)  # indexed by polygon_i
    polygons_max_overlap_class = [None] * len(polygons)

    assert isinstance(polygons, list)  # need the order constant

    for polygon_i, polygon in enumerate(polygons):
        for shape, shape_class in polygons_by_class:
            if not shape.is_valid:
                shape = shape.buffer(0)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            intersection_area = polygon.intersection(shape).area
            if intersection_area > polygons_max_overlap[polygon_i]:
                polygons_max_overlap[polygon_i] = intersection_area
                polygons_max_overlap_class[polygon_i] = shape_class

    pred_polygons_and_class = []  # include all classes

    for polygon_i, (max_overlap_area, clss) in enumerate(zip(polygons_max_overlap, polygons_max_overlap_class)):
        pred_polygons_and_class.append(
            (polygons[polygon_i], clss)
        )
    return pred_polygons_and_class, label_polygons_and_class

def get_label_and_pred_polygons_for_tile_mask_input(label_mask, path_pred_mask):
    """
    For each tile, polygonize the prediction and label mask.

    Args:
        label_mask: array that contains label mask
        path_pred_mask: path to the PNG or TIFF mask predicted by the model, where each pixel is one
            of the allowed classes.

    Returns:
        pred_polygons_and_class: list of tuples of shapely Polygon representing the geometry of the prediction,
            and the predicted class
        label_polygons_and_class: list of tuples of shapely Polygon representing the ground truth geometry,
            and the class
    """
    # polygonize the label mask
    # mask_label = np.asarray(Image.open(path_label_mask))
    label_polygons_and_class = []  # tuples of (shapely polygon, damage_class_num)
    # print('label_mask')
    # print(label_mask.shape)
    for c in all_classes:

        # default is 4-connected for connectivity
        shapes = rasterio.features.shapes(label_mask, mask=label_mask == c)

        for shape_geojson, pixel_val in shapes:
            shape = shapely.geometry.shape(shape_geojson)
            assert isinstance(shape, Polygon)
            label_polygons_and_class.append((shape, int(pixel_val)))


    # polygonize the prediction mask

    # 1. Detect the connected components by all non-background classes to determine the predicted
    # building blobs first (if we do this per class, a building with some pixels predicted to be
    # in another class will result in more buildings than connected components)
    mask_pred = np.asarray(Image.open(path_pred_mask))
    # mask_pred = cv2.medianBlur(mask_pred, 17)

    # print('mask_pred')
    # print(mask_pred.shape)
    assert len(mask_pred.shape) == 2, 'mask should be 2D only.'

    background_and_others_mask = np.where(mask_pred > 0, 1, 0).astype(np.int16)  # all non-background classes become 1

    # rasterio.features.shapes:
    # default is 4-connected for connectivity - see https://www.mathworks.com/help/images/pixel-connectivity.html
    # specify the `mask` parameter, otherwise the background will be returned as a shape
    connected_components = rasterio.features.shapes(background_and_others_mask, mask=mask_pred > 0)

    polygons = []
    for component_geojson, pixel_val in connected_components:
        # reference: https://shapely.readthedocs.io/en/stable/manual.html#python-geo-interface
        shape = shapely.geometry.shape(component_geojson)
        assert isinstance(shape, Polygon)
        if shape.area >20:
            polygons.append(shape)

    # 2. The majority class for each building blob is assigned to be that building's predicted class.
    polygons_by_class = []

    for c in all_classes:

        # default is 4-connected for connectivity
        shapes = rasterio.features.shapes(mask_pred, mask=mask_pred == c)

        for shape_geojson, pixel_val in shapes:
            shape = shapely.geometry.shape(shape_geojson)
            assert isinstance(shape, Polygon)
            polygons_by_class.append((shape, int(pixel_val)))

    # we take the class of the shape with the maximum overlap with the building polygon to be the class of the building - majority vote
    polygons_max_overlap = [0.0] * len(polygons)  # indexed by polygon_i
    polygons_max_overlap_class = [None] * len(polygons)

    assert isinstance(polygons, list)  # need the order constant

    for polygon_i, polygon in enumerate(polygons):
        for shape, shape_class in polygons_by_class:
            if not shape.is_valid:
                shape = shape.buffer(0)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            intersection_area = polygon.intersection(shape).area
            if intersection_area > polygons_max_overlap[polygon_i]:
                polygons_max_overlap[polygon_i] = intersection_area
                polygons_max_overlap_class[polygon_i] = shape_class

    pred_polygons_and_class = []  # include all classes

    for polygon_i, (max_overlap_area, clss) in enumerate(zip(polygons_max_overlap, polygons_max_overlap_class)):
        pred_polygons_and_class.append(
            (polygons[polygon_i], clss)
        )
    return pred_polygons_and_class, label_polygons_and_class
