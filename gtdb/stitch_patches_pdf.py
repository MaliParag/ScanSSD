# Author: Parag Mali
# This script stitches back the output generated on the image patches (sub-images)
# This works from PDF level detection results

# read the image
import sys
sys.path.extend(['/home/psm2208/code', '/home/psm2208/code'])
#sys.path.extend(['/media/psm2208/Workspace/ssd', '/media/psm2208/Workspace/ssd'])
import cv2
import os
import numpy as np
from multiprocessing import Pool
from cv2.dnn import NMSBoxes
from scipy.ndimage.measurements import label
from gtdb import fit_box
from gtdb import feature_extractor
import argparse
import shutil

# Default parameters for thr GTDB dataset
def parse_args():

    parser = argparse.ArgumentParser(
        description='Stitching method')

    parser.add_argument('--data_file', default='test',
                        type=str, help='choose one')
    parser.add_argument('--output_dir', default='.',
                        help='Output directory path')
    parser.add_argument('--math_dir', required=True,
                        type=str, help='detections dir')
    parser.add_argument('--math_ext', default='.csv',
                        help='Extention of detection files')
    parser.add_argument('--home_data', default='/home/psm2208/data/GTDB/', type = str,
                        help='database dir')
    parser.add_argument('--home_eval', default='/home/psm2208/code/eval/', type = str,
                        help='Eval dir')
    parser.add_argument('--home_images', default='/home/psm2208/data/GTDB/images/', type = str,
                        help='Images dir')
    parser.add_argument('--home_anno', default='/home/psm2208/data/GTDB/annotations/', type = str,
                        help='Annotations dir')
    parser.add_argument('--home_char', default='/home/psm2208/data/GTDB/char_annotations/', type = str,
                        help='Char anno dir')
    parser.add_argument('--stitching_algo', default='equal', type=str, help='Stitching algo to use')
    parser.add_argument('--algo_threshold', default=30, type=int, help='Stitching algo threshold')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--preprocess', type=bool, help='Whether to fit math regions before pooling')
    parser.add_argument('--postprocess', type=bool, help='Whether to fit math regions after pooling')

    return parser.parse_args()

def read_math(args, pdf_name):

    math_file = os.path.join(args.math_dir, pdf_name + args.math_ext)
    data = np.array([])

    if os.path.exists(math_file):
        data = np.genfromtxt(math_file, delimiter=',')

        # if there is only one entry convert it to correct form required
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

    return data

def vote_for_regions(args, math_regions, image):

    original_width = image.shape[1]
    original_height = image.shape[0]
    thresh_votes = args.algo_threshold

    votes = np.zeros(shape=(original_height, original_width))

    if args.stitching_algo == 'sum_score':
        votes = voting_sum_score(votes, math_regions)
    elif args.stitching_algo == 'max_score':
        votes = voting_max_score(votes, math_regions)
    elif args.stitching_algo == 'avg_score':
        votes = voting_avg_score(votes, math_regions)
    else:  # algorithm='equal'
        votes = voting_equal(votes, math_regions)

    # find the regions with higher than the threshold votes
    # change all the values less than thresh_votes to 0
    votes[votes < thresh_votes] = 0
    votes[votes >= thresh_votes] = 1

    return votes

def voting_sum_score(votes, math_regions):

    # cast votes for the regions
    for box in math_regions:
        votes[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = \
            votes[int(box[1]):int(box[3]), int(box[0]):int(box[2])] + box[4]

    return votes

def voting_max_score(votes, math_regions):

    # sort based on the confs. Confs is column 4
    data = math_regions[math_regions[:, 4].argsort()]

    for box in data:
        votes[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = box[4]

    return votes

def voting_equal(votes, math_regions):
    # cast votes for the regions
    for box in math_regions:
        votes[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = \
            votes[int(box[1]):int(box[3]), int(box[0]):int(box[2])] + 1

    return votes

def voting_avg_score(votes, math_regions):

    counts = np.zeros(shape=votes.shape)

    # cast votes for the regions
    for box in math_regions:
        votes[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = \
            votes[int(box[1]):int(box[3]), int(box[0]):int(box[2])] + box[4]

    # count the regions
    for box in math_regions:
        counts[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = \
            counts[int(box[1]):int(box[3]), int(box[0]):int(box[2])] + 1

    # To avoid divide by zero
    # When counts is zero, votes will be zero
    # So this should not affect the calculations and results
    counts[counts == 0] = 1

    votes = votes / counts

    return votes


def perform_nms(math_regions):

    # convert from x1,y1,x2,y2 to x,y,w,h
    math_regions[:, 2] = math_regions[:, 2] - math_regions[:, 0]
    math_regions[:, 3] = math_regions[:, 3] - math_regions[:, 1]

    scores = math_regions[:, 4]
    math_regions = np.delete(math_regions, 4, 1)

    math_regions = math_regions.tolist()
    scores = scores.tolist()

    indices = NMSBoxes(math_regions, scores, 0.2, 0.5)

    indices = [item for sublist in indices for item in sublist]
    math_regions = [math_regions[i] for i in indices]

    math_regions = np.array(math_regions)

    # restore to x1,y1,x2,y2
    math_regions[:, 2] = math_regions[:, 2] + math_regions[:, 0]
    math_regions[:, 3] = math_regions[:, 3] + math_regions[:, 1]

    return math_regions.tolist()

def preprocess_math_regions(math_regions, image):

    im_bw = convert_to_binary(image)

    preprocessed_math_regions = []

    for box in math_regions:
        box = fit_box.adjust_box(im_bw, box)
        preprocessed_math_regions.append(box)

    return preprocessed_math_regions


def voting_algo(params):

    args, math_regions, pdf_name, page_num = params
    print('Processing ', pdf_name, ' > ', page_num)

    image = cv2.imread(os.path.join(args.home_images,pdf_name,str(int(page_num+1))+".png"))

    if args.preprocess:
        math_regions = preprocess_math_regions(math_regions, image)

    # vote for the regions
    votes = vote_for_regions(args, math_regions, image)

    im_bw = convert_to_binary(image)
    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(votes, structure)

    # found the boxes. Now extract the co-ordinates left,top,right,bottom
    boxes = []
    indices = np.indices(votes.shape).T[:, :, [1, 0]]

    for i in range(ncomponents):

        labels = (labeled == (i+1))
        pixels = indices[labels.T]

        if len(pixels) < 1:
            continue

        box = [min(pixels[:, 0]), min(pixels[:, 1]), max(pixels[:, 0]), max(pixels[:, 1])]

        if args.postprocess:
            # expansion to correctly fit the region
            box = fit_box.adjust_box(im_bw, box)

        # if box has 0 width or height, do not add it in the final detections
        if feature_extractor.width(box) < 1 or feature_extractor.height(box) < 1:
            continue

        boxes.append(box)

    return boxes


def stitch(args):
    pdf_list = []
    pdf_names_file = open(args.data_file, 'r')

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    for pdf_name in pdf_names_file:
        pdf_name = pdf_name.strip()

        if pdf_name != '':
            pdf_list.append(pdf_name)

    math_regions = {}

    for pdf_name in pdf_list:
        math_regions[pdf_name] = read_math(args, pdf_name)

    voting_ip_list = []
    for pdf_name in pdf_list:

        pages = np.unique(math_regions[pdf_name][:, 0])

        #args, math_regions, pdf_name, page_num
        for page_num in pages:
            current_math = math_regions[pdf_name][math_regions[pdf_name][:,0] == page_num]
            voting_ip_list.append([args, np.delete(current_math, 0, 1), pdf_name, page_num])

    pool = Pool(processes=args.num_workers)
    out = pool.map(voting_algo, voting_ip_list)

    for ip, final_math in zip(voting_ip_list, out):

        try:
            pdf_name = ip[2]
            page_num = ip[3]

            if len(final_math) == 0:
                continue

            col = np.array([int(page_num)] * len(final_math))
            final_math = np.concatenate((col[:, np.newaxis], final_math), axis=1)

            math_file_path = os.path.join(args.output_dir, pdf_name + '.csv')

            if not os.path.exists(os.path.dirname(math_file_path)):
                os.makedirs(os.path.dirname(math_file_path))

            math_file = open(math_file_path, 'a')

            np.savetxt(math_file, final_math, fmt='%.2f', delimiter=',')
            math_file.close()
        except Exception as e:
            print("Exception while processing ", pdf_name, " ", page_num, " ", sys.exc_info(), e)

def convert_to_binary(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    im_bw = np.zeros(gray_image.shape)
    im_bw[gray_image > 127] = 0
    im_bw[gray_image <= 127] = 1

    return im_bw


if __name__ == '__main__':

    # TODO: use argparser
    args = parse_args()
    print('Using : ', args)
    stitch(args)
