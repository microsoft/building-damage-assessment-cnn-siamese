# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json

def load_json_files(json_filename):
    with open(json_filename) as f:
        file_content = json.load(f)
    return file_content

def dump_json_files(json_filename, my_dict):
    with open(json_filename, 'w') as f:
        json.dump(my_dict, f, indent=4) 
    return   
class AverageMeter(object):
    """Computes and stores the average and current value
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """

        Args:
            val: mini-batch loss or accuracy value
            n: mini-batch size
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

