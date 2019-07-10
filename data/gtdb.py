"""GTDB Dataset Classes

Original author: Parag Mali
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

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

    def __init__(self, args,
                 transform=None, target_transform=GTDBAnnotationTransform(),
                 dataset_name='GTDB'):

        self.root = args.dataset_root
        self.image_set = args.type
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.use_char_info = args.use_char_info

        #self._annopath = osp.join('%s', 'annotations', '%s.pmath')
        #self._imgpath = osp.join('%s', 'images', '%s.png')

        self._annopath = osp.join('%s', 'processed_annotations' + args.suffix, '%s.pmath')
        self._imgpath = osp.join('%s', 'processed_images' + args.suffix, '%s.png')
        self._char_annopath = osp.join('%s', 'processed_char_annotations' + args.suffix, '%s.pchar')

        self.ids = list()

        for line in open(osp.join(self.root, self.image_set)):
            self.ids.append((self.root, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w, img_id = self.pull_item(index)
        return im, gt, img_id[1]

    def __len__(self):
        return len(self.ids)


    def read_image(self, img_id):

        image = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

        if self.use_char_info:

            # First channel gray image
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image[:, :, 0] = gray_image

            second_channel = image[:, :, 1]
            third_channel = image[:, :, 2]

            second_channel[:, :] = 0
            third_channel[:, :] = 0

            char_info = np.genfromtxt((self._char_annopath % img_id), dtype=str, delimiter=',')
            if len(char_info.shape) == 1:
                char_info = char_info.reshape(1, -1)

            char_info = char_info.tolist()

            if len(char_info) > 0:
                for info in char_info:
                    if len(info) > 0:
                        # second channel - type of char using ocr code
                        second_channel[int(info[1]):int(info[3]), int(info[0]):int(info[2])] = \
                            int(info[-1][:2], 16)

                        # third channel - char label
                        third_channel[int(info[1]):int(info[3]), int(info[0]):int(info[2])] = \
                            int(info[-1][2:], 16)

        return image

    def pull_item(self, index):

        # TODO: Can not handle the case, when target is none
        img_id = self.ids[index]
        #print(img_id)

        target = self.read_annotations(self._annopath % img_id)

        img = self.read_image(img_id)

        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width, img_id
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return self.read_image(img_id)
        #return cv2.imread('/home/psm2208/Workspace/test/output.jpg', cv2.IMREAD_COLOR)

    def pull_anno(self, index, type='train'):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        gt = []

        if type == 'train':
            anno = self.read_annotations(self._annopath % img_id)
            gt = self.target_transform(anno, 1, 1)

        return img_id[1], gt

    def read_annotations(self, path):

        annotations_file = open(path, "r");

        annotations = []

        for line in annotations_file:
            line = line.split(",")
            annotations.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])

        return annotations

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
