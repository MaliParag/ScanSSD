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

    #'max_iter': 120000,
    #'feature_maps': [38, 19, 10, 5, 3, 1],
    #'min_dim': 300,

    'max_iter': 240000,
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,

    #'steps': [8, 16, 32, 64, 100, 300],
    #'min_sizes': [5, 60, 111, 162, 213, 264],
    #'max_sizes': [60, 111, 162, 213, 264, 315],

    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [8.00, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],
    'max_sizes': [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
    'aspect_ratios': [[2, 3, 5], [2, 3, 5, 7], [2, 3, 5, 7], [2, 3], [2, 3], [2], [2]],

    # 'aspect_ratios': [[2, 3, 5], [2, 3, 5, 7, 11], [2, 3, 5, 7], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'GTDB',

    'is_vertical_prior_boxes_enabled': True,
    #'is_vertical_prior_boxes_enabled': False,

    'mbox': {
        '2048': [3, 3, 3, 5, 5, 5],
        '4096': [3, 3, 3, 5, 5, 5],
        '1024': [3, 3, 3, 5, 5, 5],
        '512': [8, 10, 10, 6, 6, 4, 4],
        #'512': [5, 7, 6, 4, 4, 3, 3],
        '300': [8, 10, 10, 6, 4, 4],  # number of boxes per feature map location
    },
    'extras': {
        '2048': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        '4096': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        '1024': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
        '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    }
}


# gtdb = {
#     'num_classes': 2,
#     'lr_steps': (80000, 100000, 120000),
#     'max_iter': 120000,
#     'feature_maps': [128, 64, 32, 16, 14, 12],
#     'min_dim': 1024,
#     'steps': [8, 16, 32, 64, 100, 300],
#     'min_sizes': [5, 20, 50, 80, 120, 160],
#     'max_sizes': [20, 50, 80, 120, 160, 200],
#     'aspect_ratios': [[2], [2], [2], [2, 3, 5], [2, 3, 5], [2, 3, 5]],
#     'variance': [0.1, 0.2],
#     'clip': True,
#     'name': 'GTDB',
#     'is_vertical_prior_boxes_enabled': False,
#     'mbox': {
#         '1024': [3, 3, 3, 5, 5, 5],
#         '300': [3, 5, 5, 5, 3, 3],  # number of boxes per feature map location
#     },
#     'extras': {
#         '1024': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
#         '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
#     }
# }

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
