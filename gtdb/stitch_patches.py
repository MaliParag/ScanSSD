# Author: Parag Mali
# This script stitches back the output generated on the image patches (sub-images)

# read the image
import cv2
import os
import csv
import numpy as np
import utils.visualize as visualize
import sys
from multiprocessing import Pool
from cv2.dnn import NMSBoxes
from scipy.ndimage.measurements import label

# Default parameters for thr GTDB dataset
intermediate_width = 4800
intermediate_height = 6000
crop_size = 1200
final_width = 300
final_height = 300

stride = 1

n_horizontal = int(intermediate_width / crop_size)  # 4
n_vertical = int(intermediate_height / crop_size)  # 5

def combine_math_regions(math_files_list, image_path, output_image):

    """
    It is called for each page in the pdf
    :param math_files_list:
    :param image_path:
    :param output_image:
    :return:
    """

    image = cv2.imread(image_path)

    original_width = image.shape[1]
    original_height = image.shape[0]
    print('Read image with height, width : ', original_height, original_width)

    intermediate_width_ratio = original_width / intermediate_width
    intermediate_height_ratio = original_height / intermediate_height

    annotations_map = {}

    for math_file in math_files_list:

        name = math_file.split(os.sep)[-1]

        if os.stat(math_file).st_size == 0:
            continue

        data = np.genfromtxt(math_file, delimiter=',')

        # if there is only one entry convert it to correct form required
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        annotations_map[name] = data

    h = np.arange(0, n_horizontal - 1 + stride, stride)
    v = np.arange(0, n_vertical - 1 + stride, stride)

    for filename in annotations_map:

        data_arr = annotations_map[filename]
        patch_num = int(filename.split("_")[-1].split(".csv")[0])

        x_offset = h[(patch_num-1) % len(h)]
        y_offset = v[int((patch_num-1) / len(h))]

        if data_arr is None:
            continue

        # find scaling factors
        final_width_ratio = crop_size/final_width
        final_height_ratio = crop_size/final_height

        data_arr[:, 0] = data_arr[:, 0] * final_width_ratio
        data_arr[:, 2] = data_arr[:, 2] * final_width_ratio
        data_arr[:, 1] = data_arr[:, 1] * final_height_ratio
        data_arr[:, 3] = data_arr[:, 3] * final_height_ratio

        data_arr[:, 0] = data_arr[:, 0] + x_offset * crop_size
        data_arr[:, 2] = data_arr[:, 2] + x_offset * crop_size
        data_arr[:, 1] = data_arr[:, 1] + y_offset * crop_size
        data_arr[:, 3] = data_arr[:, 3] + y_offset * crop_size

        data_arr[:, 0] = data_arr[:, 0] * intermediate_width_ratio
        data_arr[:, 2] = data_arr[:, 2] * intermediate_width_ratio
        data_arr[:, 1] = data_arr[:, 1] * intermediate_height_ratio
        data_arr[:, 3] = data_arr[:, 3] * intermediate_height_ratio

        # multiply score by 100. Because later we convert data_arr to int datatype
        data_arr[:, 4] = data_arr[:, 4] * 100

        annotations_map[filename] = data_arr

    math_regions = np.array([])

    for key in annotations_map:

        if len(math_regions)==0:
            math_regions = annotations_map[key][:,:]
        else:
            math_regions = np.concatenate((math_regions, annotations_map[key]), axis=0)

    math_regions = math_regions.astype(int)

    #math_regions = perform_nms(math_regions)
    #math_regions = overlap_expand(math_regions)

    math_regions = math_regions.tolist()

    math_regions = voting_algo(math_regions, image)

    print(len(math_regions))

    visualize.draw_boxes_cv(image, math_regions, output_image)

    return math_regions


def voting_algo(math_regions, image, thresh_votes=5):
    """
    My voting algorithm
    :param math_regions:
    :param image:
    :return:
    """

    original_width = image.shape[1]
    original_height = image.shape[0]

    votes = np.zeros(shape=(original_height, original_width))

    # cast votes for the regions
    for box in math_regions:
        votes[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = \
            votes[int(box[1]):int(box[3]), int(box[0]):int(box[2])] + 1

    # find the regions with higher than the threshold votes
    boxes = []

    # change all the values less than thresh_votes to 0
    votes[votes < thresh_votes] = 0
    votes[votes >= thresh_votes] = 1

    # find the rectangular regions

    # this defines the connection filter
    # we allow any kind of connection
    structure = np.ones((3, 3), dtype=np.int)

    labeled, ncomponents = label(votes, structure)

    # found the boxes. Now extract the co-ordinates left,top,right,bottom

    boxes = []
    indices = np.indices(votes.shape).T[:, :, [1, 0]]

    for i in range(ncomponents):

        labels = (labeled == (i+1))
        pixels = indices[labels.T]

        box = [min(pixels[:, 0]), min(pixels[:, 1]), max(pixels[:, 0]), max(pixels[:, 1])]
        boxes.append(box)

    return boxes


def perform_nms(math_regions):

    # convert from x1,y1,x2,y2 to x,y,w,h
    math_regions[:, 2] = math_regions[:, 2] - math_regions[:, 0]
    math_regions[:, 3] = math_regions[:, 3] - math_regions[:, 1]

    scores = math_regions[:, 4]
    math_regions = np.delete(math_regions, 4, 1)

    math_regions = math_regions.tolist()
    scores = scores.tolist()

    indices = NMSBoxes(math_regions, scores, 0.2, 0.01)

    indices = [item for sublist in indices for item in sublist]
    math_regions = [math_regions[i] for i in indices]

    math_regions = np.array(math_regions)

    # restore to x1,y1,x2,y2
    math_regions[:, 2] = math_regions[:, 2] + math_regions[:, 0]
    math_regions[:, 3] = math_regions[:, 3] + math_regions[:, 1]

    return math_regions.tolist()

def overlap_expand(math_regions):

    print('Number of math regions ', len(math_regions))

    if type(math_regions) != type([]):
        math_regions = math_regions.tolist()

    obsolete = []

    for i in range(len(math_regions)):
        for j in range(i+1, len(math_regions)):
            # print(i,j)
            if intersects(math_regions[i], math_regions[j]):
                math_regions[i][0] = min(math_regions[i][0], math_regions[j][0])
                math_regions[i][1] = min(math_regions[i][1], math_regions[j][1])
                math_regions[i][2] = max(math_regions[i][2], math_regions[j][2])
                math_regions[i][3] = max(math_regions[i][3], math_regions[j][3])
                obsolete.append(j)

    math_regions = [i for j, i in enumerate(math_regions) if j not in obsolete]

    return math_regions

# check if two rectangles intersect
def intersects(first, other):
    return not (first[2] < other[0] or
                first[0] > other[2] or
                first[1] > other[3] or
                first[3] < other[1])

def stitch_patches(args):

    pdf_name, annotations_dir, output_dir, image_dir = args

    print('Processing ', pdf_name)

    #annotations_dir = '/home/psm2208/data/GTDB/processed_annotations/',
    #output_dir = '/home/psm2208/code/eval/stitched_output'
    #image_dir = '/home/psm2208/data/GTDB/images/'

    # annotation map per pdf
    # map: {'name': {'1': []}}
    annotations_map = {}

    if pdf_name not in annotations_map:
        annotations_map[pdf_name] = {}

    for root, dirs, _ in os.walk(os.path.join(annotations_dir, pdf_name), topdown=False):

        for dir in dirs:
            for filename in os.listdir(os.path.join(annotations_dir, pdf_name, dir)):

                if filename.endswith(".csv"):
                    patch_num = os.path.splitext(filename)[0]
                    page_num = os.path.basename(os.path.join(annotations_dir, pdf_name, dir))

                    if page_num not in annotations_map[pdf_name]:
                        annotations_map[pdf_name][page_num] = []

                    annotations_map[pdf_name][page_num].append(os.path.join(annotations_dir, pdf_name, dir, filename))

    math_file = open(os.path.join(output_dir, pdf_name+'.csv'),'w')

    writer = csv.writer(math_file, delimiter=",")

    if not os.path.exists(os.path.join(output_dir, pdf_name)):
        os.mkdir(os.path.join(output_dir, pdf_name))

    for key in sorted(annotations_map[pdf_name]):

        # basically it is called for each page in the pdf
        math_regions = combine_math_regions(
                        annotations_map[pdf_name][key],
                        os.path.join(image_dir, pdf_name, key + '.png'),
                        os.path.join(output_dir, pdf_name, key + '.png'))

        for math_region in math_regions:
            math_region.insert(0,int(key)-1)
            writer.writerow(math_region)

    math_file.close()


def patch_stitch(filename, annotations_dir, output_dir, image_dir='/home/psm2208/data/GTDB/images/'):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    training_pdf_names_list = []
    training_pdf_names = open(filename, 'r')

    for pdf_name in training_pdf_names:
        pdf_name = pdf_name.strip()

        if pdf_name != '':
            training_pdf_names_list.append((pdf_name, annotations_dir, output_dir, image_dir))

    training_pdf_names.close()

    pool = Pool(processes=4)
    pool.map(stitch_patches, training_pdf_names_list)
    pool.close()
    pool.join()


if __name__ == '__main__':
    #stitch_patches("TMJ_1990_163_193")
    #stitch_patches("AIF_1970_493_498", "/home/psm2208/code/eval")
    #filename = sys.argv[1] # train_pdf

    #stitch_patches(("Arkiv_1997_185_199", "/home/psm2208/code/eval/Train_Focal_100",
    #             "/home/psm2208/code/eval/Train_Focal_100/out", '/home/psm2208/data/GTDB/images/'))

    #patch_stitch(filename, sys.argv[2], sys.argv[3])
    stride = 0.1
    patch_stitch("/home/psm2208/data/GTDB/train_pdf", "/home/psm2208/code/eval/Train_Focal_10_25",
                 "/home/psm2208/code/eval/Train_Focal_10_25/none", '/home/psm2208/data/GTDB/images/')
