'''
This file contains functions to test and save the results
'''
from __future__ import print_function
import os
import argparse
import torch.backends.cudnn as cudnn
from ssd import build_ssd
from utils import draw_boxes, helpers, save_boxes
import logging
import time
import datetime
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from data import *
import shutil
import torch.nn as nn

def test_net_batch(args, net, gpu_id, dataset, transform, thresh):
    '''
    Batch testing
    '''
    num_images = len(dataset)

    if args.limit != -1:
        num_images = args.limit

    data_loader = DataLoader(dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=False, collate_fn=detection_collate,
                              pin_memory=True)

    total = len(dataset)

    logging.debug('Test dataset size is {}'.format(total))

    done = 0

    for batch_idx, (images, targets, metadata) in enumerate(data_loader):

        done = done + len(images)
        logging.debug('processing {}/{}'.format(done, total))

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        y, debug_boxes, debug_scores = net(images)  # forward pass
        detections = y.data

        k = 0
        for img, meta in zip(images, metadata):

            img_id = meta[0]
            x_l = meta[1]
            y_l = meta[2]

            img = img.permute(1,2,0)
            # scale each detection back up to the image
            scale = torch.Tensor([img.shape[1], img.shape[0],
                                  img.shape[1], img.shape[0]])

            recognized_boxes = []
            recognized_scores = []

            # [1,2,200,5]
            # we only care about math class
            # hence select detections[image_id, class, detection_id, detection_score]
            # class=1 for math
            i = 1
            j = 0

            while j < detections.size(2) and detections[k, i, j, 0] >= thresh:  # TODO it was 0.6

                score = detections[k, i, j, 0]
                pt = (detections[k, i, j, 1:] * args.window).cpu().numpy()
                coords = (pt[0] + x_l, pt[1] + y_l, pt[2] + x_l, pt[3] + y_l)
                #coords = (pt[0], pt[1], pt[2], pt[3])
                recognized_boxes.append(coords)
                recognized_scores.append(score.cpu().numpy())

                j += 1

            save_boxes(args, recognized_boxes, recognized_scores, img_id)
            k = k + 1

            if args.verbose:
                draw_boxes(args, img.cpu().numpy(), recognized_boxes, recognized_scores,
                           debug_boxes, debug_scores, scale, img_id)

def test_gtdb(args):

    gpu_id = 0
    if args.cuda:
        gpu_id = helpers.get_freer_gpu()
        torch.cuda.set_device(gpu_id)

    # load net
    num_classes = 2 # +1 background

    # initialize SSD
    net = build_ssd(args, 'test', exp_cfg[args.cfg], gpu_id, args.model_type, num_classes)

    logging.debug(net)
    net.to(gpu_id)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(args.trained_model, map_location={'cuda:1':'cuda:0'}))
    net.eval()
    logging.debug('Finished loading model!')

    dataset = GTDBDetection(args, args.test_data, split='test',
                            transform=BaseTransform(args.model_type, (246,246,246)),
                            target_transform=GTDBAnnotationTransform())

    if args.cuda:
        net = net.to(gpu_id)
        cudnn.benchmark = True

    # evaluation
    test_net_batch(args, net, gpu_id, dataset,
                   BaseTransform(args.model_type, (246,246,246)),
                   thresh=args.visual_threshold)

def parse_args():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
    parser.add_argument('--trained_model', default='weights/ssd300_GTDB_990.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--save_folder', default='eval/', type=str,
                        help='Dir to save results')
    parser.add_argument('--visual_threshold', default=0.6, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--cuda', default=False, type=bool,
                        help='Use cuda to train model')
    parser.add_argument('--dataset_root', default=GTDB_ROOT, help='Location of VOC root directory')
    parser.add_argument('--test_data', default="testing_data", help='testing data file')
    parser.add_argument('--verbose', default=False, type=bool, help='plot output')
    parser.add_argument('--suffix', default="_10", type=str, help='suffix of directory of images for testing')
    parser.add_argument('--exp_name', default="SSD", help='Name of the experiment. Will be used to generate output')
    parser.add_argument('--model_type', default=300, type=int,
                        help='Type of ssd model, ssd300 or ssd512')
    parser.add_argument('--use_char_info', default=False, type=bool, help='Whether or not to use char info')
    parser.add_argument('--limit', default=-1, type=int, help='limit on number of test examples')
    parser.add_argument('--cfg', default="gtdb", type=str,
                        help='Type of network: either gtdb or math_gtdb_512')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in data loading')
    parser.add_argument('--kernel', default="3 3", type=int, nargs='+',
                        help='Kernel size for feature layers: 3 3 or 1 5')
    parser.add_argument('--padding', default="1 1", type=int, nargs='+',
                        help='Padding for feature layers: 1 1 or 0 2')
    parser.add_argument('--neg_mining', default=True, type=bool,
                        help='Whether or not to use hard negative mining with ratio 1:3')
    parser.add_argument('--log_dir', default="logs", type=str,
                        help='dir to save the logs')
    parser.add_argument('--stride', default=0.1, type=float,
                        help='Stride to use for sliding window')
    parser.add_argument('--window', default=1200, type=int,
                        help='Sliding window size')

    parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")

    args = parser.parse_args()

    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if os.path.exists(os.path.join(args.save_folder, args.exp_name)):
        shutil.rmtree(os.path.join(args.save_folder, args.exp_name))

    return args

if __name__ == '__main__':

    args = parse_args()
    start = time.time()
    try:
        filepath=os.path.join(args.log_dir, args.exp_name + "_" + str(round(time.time())) + ".log")
        print('Logging to ' + filepath)
        logging.basicConfig(filename=filepath,
                            filemode='w', format='%(process)d - %(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

        test_gtdb(args)
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

    end = time.time()
    logging.debug('Toal time taken ' + str(datetime.timedelta(seconds=end-start)))
    logging.debug("Testing done!")

