from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, args, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        self.is_vertical_prior_boxes_enabled = cfg['is_vertical_prior_boxes_enabled']
        self.args = args
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []

        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]

                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]

                    if self.is_vertical_prior_boxes_enabled:
                        mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


if __name__ == '__main__':

    pass
    # [2, 3, 5, 7, 10], [2, 3, 5, 7, 10], [2, 3, 5, 7, 10],
    # [2, 3, 5, 7, 10], [2, 3, 5, 7, 10], [2, 3, 5, 7, 10]

    #test prior box
    # import matplotlib
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches
    # import os
    # import numpy as np
    # import cv2
    #
    # img = np.zeros((512,512))
    #
    # boxes = {}
    #
    # num_classes = 2
    # feature_maps = [64, 32, 16, 8, 4, 2, 1]
    # image_size = 512
    # steps = [8, 16, 32, 64, 128, 256, 512]
    # min_sizes = [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8]
    # max_sizes = [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6]
    #
    # #aspect_ratios = [[2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10]]
    # aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2,3], [2], [2]]
    #
    # is_vertical_prior_boxes_enabled = True
    #
    # image_path = '/home/psm2208/data/GTDB/processed_images_512/Erbe94/3/47.png'
    # image = cv2.imread(image_path)
    # #image = np.array(image, dtype=float)
    #
    # box_centers={}
    # current_center = {}
    #
    # for k, f in enumerate(feature_maps):
    #     mean = []
    #     centers = []
    #     for i, j in product(range(f), repeat=2):
    #         f_k = image_size / steps[k]
    #         # unit center x,y
    #         cx = (j + 0.5) / f_k
    #         cy = (i + 0.5) / f_k
    #
    #         centers.append([cx,cy])
    #
    #     box_centers[f] = centers
    #
    #     # for i in range(f):
    #     #     j = int(f/2) #TODO
    #     #     i = int(f/2)
    #     for i, j in product(range(f), repeat=2):
    #         i = 17
    #         j = 15
    #
    #         f_k = image_size / steps[k]
    #         # unit center x,y
    #         cx = (j + 0.5) / f_k
    #         cy = (i + 0.5) / f_k
    #
    #         current_center[f] = [cx, cy]
    #         # aspect_ratio: 1
    #         # rel size: min_size
    #         s_k = min_sizes[k] / image_size
    #         mean += [cx, cy, s_k, s_k]
    #
    #         # aspect_ratio: 1
    #         # rel size: sqrt(s_k * s_(k+1))
    #         s_k_prime = sqrt(s_k * (max_sizes[k] / image_size))
    #         mean += [cx, cy, s_k_prime, s_k_prime]
    #
    #         # rest of aspect ratios
    #         for ar in aspect_ratios[k]:
    #             mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
    #
    #             if is_vertical_prior_boxes_enabled:
    #                 mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
    #
    #         break #TODO: remove
    #
    #     output = torch.Tensor(mean).view(-1, 4)
    #     boxes[f] = output
    #
    #
    # colors = ['r','g','y','m','c','#d6a920','b']
    #
    # #fig2, ax2 = plt.subplots(1)
    # #fig, ax = plt.subplots(1)
    #
    # for k,f in enumerate(feature_maps):
    #     fig, ax = plt.subplots(1)
    #
    #     output = boxes[f] * image_size
    #     #output[:, :2] = output[:, :2] * image_size
    #     #output[:, 2:] = output[:, 2:] * steps[k]
    #
    #     for box in output:
    #         width = box[2]
    #         height = box[3]
    #
    #         x = np.round(max(width/height, height/width)).numpy()
    #
    #         rect = patches.Rectangle((box[0] - width/2, box[1] - height/2), width, height,
    #                                  linewidth=2, edgecolor=colors[int(x-1)], facecolor='none', alpha=1.0)
    #         # Add the patch to the Axes
    #         ax.add_patch(rect)
    #
    #     centers=np.array(box_centers[f]) * image_size
    #     plt.scatter(centers[:,0], centers[:,1], s=7, marker='*', color='b', alpha=0.4)
    #     plt.scatter(current_center[f][0]*image_size, current_center[f][1]*image_size,s=10,
    #                 marker='*', color='#5bfc70')
    #
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #
    #     ax.imshow(img)
    #     plt.title("All default boxes for {} x {} feature map".format(f, f))
    #     plt.savefig('/home/psm2208/code/default_boxes/colors_all_default_box_vis_without_img_{}.png'.format(f), dpi=600)
    #
    #     ax.imshow(image)
    #     plt.savefig('/home/psm2208/code/default_boxes/colors_all_default_box_vis_{}.png'.format(f), dpi=600)
    #
    #     plt.clf()

    # ax.set_xticks([])
    # ax.set_yticks([])
    #
    # ax.imshow(img)
    # plt.title("All default boxes for SSD512")
    # plt.savefig('/home/psm2208/code/default_boxes/all_colors_default_boxes_without_img.png', dpi=600)
    #
    # ax.imshow(image)
    # plt.savefig('/home/psm2208/code/default_boxes/all_colors_default_boxes.png', dpi=600)
    #
    # plt.close()