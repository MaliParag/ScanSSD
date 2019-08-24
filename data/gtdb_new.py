"""
Author: Parag Mali
Data reader for the GTDB dataset
Uses sliding windows to generate sub-images
"""

from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from gtdb import box_utils
from gtdb import feature_extractor
import copy
import utils.visualize as visualize


GTDB_CLASSES = (  # always index 0 is background
    'math')

GTDB_ROOT = osp.join(HOME, "data/GTDB/")


class GTDBAnnotationTransform(object):
    """Transforms a GTDB annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None):
        pass

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotations. This will be the list of bounding boxes
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []

        # read the annotations
        for box in target:
            res.append([box[0]/width, box[1]/height, box[2]/width, box[3]/height, 0])

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class GTDBDetection(data.Dataset):
    """GTDB Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to GTDB folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name: `GTDB`
    """


    def __init__(self, args, data_file, split='train',
                 transform=None, target_transform=GTDBAnnotationTransform(),
                 dataset_name='GTDB'):

        #split can be train, validate or test

        self.root = args.dataset_root
        self.image_set = data_file
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.use_char_info = args.use_char_info

        # The stride to use for the windowing
        self.stride = args.stride # 0.1
        self.window = args.window # 1200

        self._annopath = osp.join('%s', 'annotations', '%s.pmath')
        self._imgpath = osp.join('%s', 'images', '%s.png')
        self._char_annopath = osp.join('%s', 'char_annotations', '%s.pchar')

        self.ids = list()

        for line in open(osp.join(self.root, self.image_set)):
            self.ids.append((self.root, line.strip()))

        # initialize the training images and annotations
        self.images = {}
        self.math_ground_truth = {}
        self.is_math = {}
        # for each id store the image patch which has math
        # This can be used as a box util
        self.metadata = []

        self.read_all_images()
        self.read_gt_annotations()
        self.generate_metadata()

    def generate_metadata(self):

        for id in self.ids:
            math_patches = []

            height, width, channels = self.images[id[1]].shape
            current_page_boxes = self.math_ground_truth[id[1]]
            n_horizontal = np.ceil(width / self.window)  # 4
            n_vertical = np.ceil(height / self.window)  # 5

            h = np.arange(0, n_horizontal - 1 + self.stride, self.stride)
            v = np.arange(0, n_vertical - 1 + self.stride, self.stride)
            crop_size = self.window

            if (self.split == 'train' or self.split == 'validate')  and self.is_math[id[1]]:

                for i in h:
                    for j in v:
                        x_l = int(np.round(crop_size * i))
                        x_h = x_l + self.window

                        y_l = int(np.round(crop_size * j))
                        y_h = y_l + self.window

                        # left, top, right, bottom
                        image_box = [x_l, y_l, x_h, y_h]

                        current_page_boxes = copy.deepcopy(self.math_ground_truth[id[1]])

                        # if math intersects only consider the region which
                        # is part of the current bounding box
                        for box in current_page_boxes:
                            if box_utils.intersects(image_box, box):
                                # left, top, right, bottom
                                # y increases downwards

                                # crop the boxes to fit into image region
                                box[0] = max(x_l, box[0])
                                box[1] = max(y_l, box[1])
                                box[2] = min(x_h, box[2])
                                box[3] = min(y_h, box[3])

                                # # Translate to origin
                                box[0] = box[0] - x_l
                                box[2] = box[2] - x_l

                                box[1] = box[1] - y_l
                                box[3] = box[3] - y_l

                                if feature_extractor.width(box) > 0 and feature_extractor.height(box) > 0:
                                    self.metadata.append([id[1], x_l, y_l])
                                    break
            elif self.split=='test':
                for i in h:
                    for j in v:
                        x_l = int(np.round(crop_size * i))
                        y_l = int(np.round(crop_size * j))
                        self.metadata.append([id[1], x_l, y_l])

    def read_all_images(self):

        # This function reads all the images in the training dataset.
        # GTDB dataset is small, loading all the images at once is not a problem

        for id in self.ids:
            image = cv2.imread(self._imgpath % id, cv2.IMREAD_COLOR)
            self.images[id[1]] = image

    def read_gt_annotations(self):

        # This function reads all the annotations for the training images
        for id in self.ids:
            if osp.exists(self._annopath % id):
                gt_regions = np.genfromtxt(self._annopath % id, delimiter=',')
                gt_regions = gt_regions.astype(int)

                # if there is only one entry convert it to correct form required
                if len(gt_regions.shape) == 1:
                    gt_regions = gt_regions.reshape(1, -1)

                self.math_ground_truth[id[1]] = gt_regions
                self.is_math[id[1]] = True
            else:
                self.math_ground_truth[id[1]] = np.array([-1,-1,-1,-1]).reshape(1,-1)
                self.is_math[id[1]] = False

    def __getitem__(self, index):
        im, gt, metadata = self.pull_item(index)
        return im, gt, metadata

    def __len__(self):
        return len(self.metadata)

    def gen_targets(self, index):

        metadata = self.metadata[index]

        x_l = metadata[1]
        y_l = metadata[2]
        x_h = x_l + self.window
        y_h = y_l + self.window

        current_page_boxes = copy.deepcopy(self.math_ground_truth[metadata[0]])
        targets = []

        image_box = [x_l, y_l, x_h, y_h]

        # if math intersects only consider the region which
        # is part of the current bounding box
        for box in current_page_boxes:
            if box_utils.intersects(image_box, box):
                # left, top, right, bottom
                # y increases downwards

                # crop the boxes to fit into image region
                box[0] = max(x_l, box[0])
                box[1] = max(y_l, box[1])
                box[2] = min(x_h, box[2])
                box[3] = min(y_h, box[3])

                # # Translate to origin
                box[0] = box[0] - x_l
                box[2] = box[2] - x_l

                box[1] = box[1] - y_l
                box[3] = box[3] - y_l

                if feature_extractor.width(box) > 0 and feature_extractor.height(box) > 0:
                    targets.append(box)

        # It is done only for testing, where we do not care about targets
        # This avoids IndexError: too many indices for array
        # TODO: refactor in future
        if len(targets) == 0:
            targets = [[-1,-1,-1,-1]]

        return targets

    def gen_image(self, index):

        metadata = self.metadata[index]
        image = self.images[metadata[0]]

        x_l = metadata[1]
        y_l = metadata[2]
        x_h = x_l + min(self.window, image.shape[1]-x_l)
        y_h = y_l + min(self.window, image.shape[0]-y_l)

        cropped_image = np.full((self.window, self.window, image.shape[2]), 255)
        cropped_image[:y_h-y_l, :x_h-x_l, :] = image[y_l: y_h, x_l: x_h, :]

        return cropped_image

    def pull_item(self, index):

        metadata = self.metadata[index]

        target = self.gen_targets(index)
        img = self.gen_image(index)

        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))


        return torch.from_numpy(img).permute(2, 0, 1), target, metadata
