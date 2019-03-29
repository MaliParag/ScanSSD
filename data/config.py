# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
    'mbox': {
         '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
         '512': [],
    },
    'extras': {
        '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        '512': [],
    }
}

gtdb = {
    'num_classes': 2,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [128, 64, 32],
    'min_dim': 1024,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [10, 40, 100],
    'max_sizes': [40, 100, 300],
    'aspect_ratios': [[2], [2, 3, 5], [2, 3, 5]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'GTDB',
    'is_vertical_prior_boxes_enabled': False,
    'mbox': {
        '1024': [3, 5, 5],
        '300': [3, 5, 5],  # number of boxes per feature map location
    },
    'extras': {
        '1024': [256, 'S', 512, 128],
        '300': [256, 'S', 512, 128],
    }
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
    'mbox': {
         '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
         '512': [],
    },
    'extras': {
        '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        '512': [],
    }
}
