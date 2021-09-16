# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Data processing

## Generate masks from polygons

We generate pixel masks based on the xBD dataset labels provided as polygons in geoJSON files, since the tier3 disasters did not come with masks and the masks for the other disasters had a border value that was likely 0, which would not help to separate the buildings. 

We modified the xView baseline repo's [script](https://github.com/DIUx-xView/xView2_baseline/blob/master/utils/mask_polygons.py) for `create_label_masks.py` to generate the masks for all wind disasters. Running the script only took < 10 minutes. Commands that we ran:
```
python data/create_label_masks.py /home/lynx/mnt/nlrc-damage-assessment/public_datasets/xBD/raw/hold -b 1

python data/create_label_masks.py /home/lynx/mnt/nlrc-damage-assessment/public_datasets/xBD/raw/test -b 1

python data/create_label_masks.py /home/lynx/mnt/nlrc-damage-assessment/public_datasets/xBD/raw/train -b 1

python data/create_label_masks.py /home/lynx/mnt/nlrc-damage-assessment/public_datasets/xBD/raw_tier3 -b 1

python data/create_label_masks.py /home/lynx/mnt/nlrc-damage-assessment/public_datasets/xBD/raw/hold -b 2

python data/create_label_masks.py /home/lynx/mnt/nlrc-damage-assessment/public_datasets/xBD/raw/test -b 2

python data/create_label_masks.py /home/lynx/mnt/nlrc-damage-assessment/public_datasets/xBD/raw/train -b 2

python data/create_label_masks.py /home/lynx/mnt/nlrc-damage-assessment/public_datasets/xBD/raw_tier3 -b 2

```

Masks for border width of 1 and 2 were created in case we would like to experiment. You can see their effects in the notebook [inspect_masks.ipynb](./inspect_masks.ipynb).
