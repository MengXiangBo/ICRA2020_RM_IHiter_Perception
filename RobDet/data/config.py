# config.py
import os.path

# For slim-yolov2
ANCHOR_SIZE = [[1.19, 2.14], [1.19, 2.14], [2.15, 3.81], [3.96, 6.19], [7.16, 10.03]]

# For tiny-yolov3
MULTI_ANCHOR_SIZE = [[16.48, 29.40], [25.74, 48.39], [43.01, 77.63],
                     [71.91, 125.20], [127.39, 199.38], [229.54, 321.77]]

IGNORE_THRESH = 0.5

# RM CONFIGS
rm_ab = {
    'num_classes': 2,
    'lr_epoch': (30, 60),
    'max_epoch': 100,
    'min_dim': [608, 608],
    'name': 'RM',
}
