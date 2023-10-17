#!/usr/bin/env bash

docker run \
    --name "nlrc-model" \
    --rm \
    -v /datadrive/nlrc:/mnt \
    nlrc-building-damage-assessment:latest \
    "--output_dir" "/mnt" \
    "--data_img_dir" "/mnt/dataset" \
    "--data_inference_dict" "/mnt/constants/splits/all_disaster_splits_sliced_img_augmented_20.json" \
    "--data_mean_stddev" "/mnt/constants/splits/all_disaster_mean_stddev_tiles_0_1.json" \
    "--label_map_json" "/mnt/constants/class_lists/xBD_label_map.json" \
    "--model" "/mnt/models/model_best.pth.tar"
