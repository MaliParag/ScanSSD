# Author: Parag Mali
# This script stitches back the output generated on the image patches (sub-images)

# read the image
import sys
sys.path.extend(['/home/psm2208/code', '/home/psm2208/code'])
import cv2
import os
import csv
import numpy as np
import utils.visualize as visualize
from multiprocessing import Pool
from cv2.dnn import NMSBoxes
from scipy.ndimage.measurements import label
import scipy.ndimage as ndimage


# Default parameters for thr GTDB dataset
intermediate_width = 4800
intermediate_height = 6000
crop_size = 1200
final_width = 300
final_height = 300

stride = 1

n_horizontal = int(intermediate_width / crop_size)  # 4
n_vertical = int(intermediate_height / crop_size)  # 5
algorithm = 'equal'

def combine_math_regions(pdf_name, page_num, math_files_list, image_path, output_image,
                         gt_dir, thresh):

    """
    It is called for each page in the pdf
    :param math_files_list:
    :param image_path:
    :param output_image:
    :return:
    """

    print(image_path)
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

    # intital math regions
    #math_regions_initial = np.copy(math_regions)

    math_regions = preprocess_math_regions(math_regions, image)
    processed_math_regions = np.copy(math_regions)

    #math_regions = perform_nms(math_regions)
    #math_regions = overlap_expand(math_regions)

    #math_regions = math_regions.tolist()

    # This will give final math regions
    #math_regions = voting_algo(math_regions, image, algorithm='equal', thresh_votes=thresh)
    #math_regions = voting_algo(math_regions, image, algorithm='max_score', thresh_votes=thresh)
    math_regions = voting_algo(math_regions, image, algorithm=algorithm, thresh_votes=thresh)
    #math_regions = voting_algo(math_regions, image, algorithm='avg_score', thresh_votes=thresh)

    print(len(math_regions))

    gt_regions = None
    #gt_path = os.path.join(gt_dir, pdf_name, page_num + ".pmath")
    #gt_regions = np.genfromtxt(gt_path, delimiter=',')
    #gt_regions = gt_regions.astype(int)

    # if there is only one entry convert it to correct form required
    #if len(gt_regions.shape) == 1:
    #   gt_regions = gt_regions.reshape(1, -1)

    #gt_regions = gt_regions.tolist()

    visualize.draw_all_boxes(image, processed_math_regions, math_regions, gt_regions, output_image)
    print('saved ', output_image)
    # visualize.draw_boxes_cv(image, math_regions, gt_regions, output_image)
    # Following works only with 'none' stitching method
    # Can't use any stitching method with this function
    #visualize.draw_stitched_boxes(image, math_regions, output_image)

    return math_regions

def preprocess_math_regions(math_regions, image):

    im_bw = convert_to_binary(image)

    args = []

    for box in math_regions:
        args.append((im_bw, box))
        #preprocessed_math_regions.append(box)

    pool = Pool(processes=24)
    preprocessed_math_regions = pool.map(adjust_box, args)
    pool.close()
    pool.join()


    return preprocessed_math_regions


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

def voting_sum_score(votes, math_regions):

    # cast votes for the regions
    for box in math_regions:
        votes[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = \
            votes[int(box[1]):int(box[3]), int(box[0]):int(box[2])] + box[4]

    return votes

def voting_heuristic_score(votes, math_regions):

    # All the connected components should have equal score

    # Threshold on the score

    # Threshold on the votes
    pass

def voting_max_score(votes, math_regions):

    # sort based on the confs. Confs is column 4
    data = math_regions[math_regions[:, 4].argsort()]

    for box in data:
        votes[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = box[4]

    return votes

def vote_for_regions(math_regions, image, algorithm, thresh_votes):

    original_width = image.shape[1]
    original_height = image.shape[0]

    votes = np.zeros(shape=(original_height, original_width))

    if algorithm == 'sum_score':
        thresh_votes = thresh_votes * 100
        votes = voting_sum_score(votes, math_regions)
    elif algorithm == 'max_score':
        votes = voting_max_score(votes, math_regions)
    elif algorithm == 'avg_score':
        thresh_votes = thresh_votes * 100
        votes = voting_avg_score(votes, math_regions)
    else:  # algorithm='equal'
        votes = voting_equal(votes, math_regions)

    # find the regions with higher than the threshold votes
    # change all the values less than thresh_votes to 0
    votes[votes < thresh_votes] = 0
    votes[votes >= thresh_votes] = 1

    return votes

def label_regions(math_regions, image):
    labeled = np.zeros(image.shape[:2])

    math_regions = math_regions[math_regions[:, 4].argsort()]

    for label, math_region in enumerate(math_regions):
        labeled[math_region[1]:math_region[3], math_region[0]:math_region[2]] = label

    #uniq_labels = np.unique(labeled)

    return labeled

def voting_algo(math_regions, image, algorithm='equal', thresh_votes=20):

    # vote for the regions
    votes = vote_for_regions(math_regions, image, algorithm, thresh_votes)
    #math_region_labels = label_regions(math_regions, image)

    votes_regions = np.copy(votes)
    #math_regions = math_regions[math_regions[:, 4].argsort()]

    # TODO : Row info might be useful
    # use image to find the rows where all values are zero
    # use that info separate lines
    # so, that math regions on different lines do not get merged
    # This allows for horizontal merging, but skips vertical merging
    #

    #blank_rows = find_blank_rows(image, 0)

    # for blank rows, zero votes
    #for box in blank_rows:
    #    votes_regions[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 0

    # blank_rows = find_blank_rows_h(image)
    #
    # # for blank rows, zero votes
    # for row in blank_rows:
    #     votes_regions[row] = 0
    #     #image[row] = 0

    #cv2.imwrite('/home/psm2208/test.png', image)

    # this defines the connection filter
    # we allow any kind of connection
    # votes = votes.astype(np.uint8)
    #
    # contours, _ = cv2.findContours(votes, 1, 2)
    #
    # for cnt in contours:
    #     approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    # #    print('len ', len(approx))
    #     if len(approx) >= 4:
    #         #print("rectangle")
    #         #cv2.drawContours(image, [cnt], 0, 255, -1)
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         votes[y:y+h, x:x+w] = votes_regions[y:y+h, x:x+w]

    # cv2.imwrite("test.png", image)

    im_bw = convert_to_binary(image)

    structure = np.ones((3, 3), dtype=np.int)

    # find the rectangular regions
    # votes = ndimage.binary_opening(votes, structure=structure)

    labeled, ncomponents = label(votes, structure)

    # found the boxes. Now extract the co-ordinates left,top,right,bottom
    boxes = []
    indices = np.indices(votes.shape).T[:, :, [1, 0]]

    for i in range(ncomponents):

        labels = (labeled == (i+1))
        #labels = (labeled == i)
        pixels = indices[labels.T]

        if len(pixels) < 1:
            continue

        box = [min(pixels[:, 0]), min(pixels[:, 1]), max(pixels[:, 0]), max(pixels[:, 1])]

        # put all ones in bw image
        #cv2.imwrite("/home/psm2208/test.png", im_bw * 255)

        # expansion to correctly fit the region
        box = adjust_box((im_bw, box))
        # if not np.all(votes[box[1]:box[3], box[0]:box[2]]):
        #
        #     ccs = np.unique(math_region_labels[box[1]:box[3], box[0]:box[2]])
        #
        #     # As the math regions are labels from low confidence to high confidence
        #     # the last region in ccs is the highest confident region.
        #     #TODO Need to apply NMS during this step
        #     # if 50% overlap reject the box
        #     # once NMS is done, do overlap-expand
        #     # if it is horizontal overlap, check if horizontal edge of larger rectangle
        #     # has more than 25% overlap then expand, otherwise do not expand
        #     # Finally do cc edge correction
        #     # Also, if horizontally very close merge the regions
        #     regions = []
        #     for cc in ccs:
        #         regions.append(math_regions[int(cc)])
        #
        #     regions = perform_nms(np.array(regions))
        #     boxes.extend(regions)
        #
        # else:
        #     boxes.append(box)
        boxes.append(box)

    return boxes

def adjust_box(args):
    im_bw, box = args
    box = expand(im_bw, box)
    box = contract(im_bw, box)
    return box

def contract(im_bw, box):

    # find first row with one pixel
    rows_with_pixels = np.any(im_bw[box[1]:box[3], box[0]:box[2]], axis=1)
    cols_with_pixels = np.any(im_bw[box[1]:box[3], box[0]:box[2]], axis=0)

    if len(rows_with_pixels==True) == 0 or len(cols_with_pixels==True) == 0:
        box = [0,0,0,0,0]
        return box

    left = box[0] + np.argmax(cols_with_pixels==True)
    top = box[1] + np.argmax(rows_with_pixels==True)
    right = box[0] + len(cols_with_pixels) - np.argmax(cols_with_pixels[::-1]==True) - 1
    bottom = box[1] + len(rows_with_pixels) - np.argmax(rows_with_pixels[::-1]==True) - 1

    box[0] = left
    box[1] = top
    box[2] = right
    box[3] = bottom

    return box

    # find first column with one pixel
    # find last row with one pixel
    # find last col with pixel

def expand(im_bw, box):

    im_copy = np.copy(im_bw)
    im_copy[box[1]:box[3], box[0]:box[2]] = 1

    start = (box[1], box[0])
    queue = [start]
    visited = set()

    while len(queue) != 0:
        front = queue.pop(0)
        if front not in visited:
            for adjacent_space in get_adjacent_spaces(im_copy, front, visited):
                queue.append(adjacent_space)

            box[0] = min(front[1], box[0]) #left
            box[1] = min(front[0], box[1]) #top
            box[2] = max(front[1], box[2])  # left + width
            box[3] = max(front[0], box[3])  # top + height

            visited.add(front)

    return box

def get_adjacent_spaces(im_bw, space, visited):

    spaces = list()
    dirs = [[1,0],[-1,0],[0,1],[0,-1]]

    for dir in dirs:
        r = space[0] + dir[0]
        c = space[1] + dir[1]

        if r < im_bw.shape[0] and c < im_bw.shape[1] and r >= 0 and c >= 0:
            spaces.append((r, c))

    final = list()
    for i in spaces:
        if im_bw[i[0]][i[1]] == 1 and i not in visited:
            final.append(i)

    return final

def convert_to_binary(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    im_bw = np.zeros(gray_image.shape)
    im_bw[gray_image > 127] = 0
    im_bw[gray_image <= 127] = 1

    return im_bw

def find_blank_rows_h(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    im_bw = np.zeros(gray_image.shape)
    im_bw[gray_image > 127] = 0
    im_bw[gray_image <= 127] = 1

    row_sum = np.sum(im_bw, axis=1)

    cum_sum = np.zeros(row_sum.shape)

    cum_sum[0] = row_sum[0]

    for i, sum in enumerate(row_sum[1:]):
        cum_sum[i+1] = cum_sum[i] + sum

    blank_rows = []
    for i, sum in enumerate(cum_sum):
        if is_blank(cum_sum, i):
            blank_rows.append(i)

    return blank_rows

# check n last rows
def is_blank(cum_sum, current, n=30, thresh=3000):

    # It is not a blank row
    ret = False

    # check above
    # if (current > 0) and ((cum_sum[current] - cum_sum[current-1]) == 0):
    #
    #     a_thresh = thresh
    #     if current < n:
    #         val = cum_sum[current]
    #         a_thresh = ((thresh/n) * (current + 1))
    #     else:
    #         val = cum_sum[current] - cum_sum[current-n]
    #
    #     # It is  a blank row
    #     if val >= a_thresh:
    #         ret = True

    # check below
    if (current < len(cum_sum)) and (cum_sum[current] - cum_sum[current-1]) == 0:

        b_thresh = thresh

        if current + n >= len(cum_sum):
            val = cum_sum[len(cum_sum)-1] - cum_sum[current]
            b_thresh = (thresh/n) * (len(cum_sum) - current)
        else:
            val = cum_sum[current + n] - cum_sum[current]

        # It is  a blank row
        if val >= b_thresh:
            ret = True

    return ret


def find_blank_rows(image, line_spacing=15):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blank_rows = np.all(gray_image == 255, axis=1)

    im_bw = np.zeros(gray_image.shape)
    im_bw[blank_rows] = 255
    #gray_image[~blank_rows] = 0

    #cv2.imwrite("/home/psm2208/code/eval/test.png", im_bw)

    labeled, ncomponents = ndimage.label(im_bw)
    rows = []

    indices = np.indices(im_bw.shape).T[:, :, [1, 0]]

    # for i in range(ncomponents):
    #
    #     labels = (labeled == (i+1))
    #     pixels = indices[labels.T]
    #
    #     box = [min(pixels[:, 0]), min(pixels[:, 1]), max(pixels[:, 0]), max(pixels[:, 1])]
    #
    #     if box[2] - box[0] > line_spacing:
    #         rows.append(box)

    line_bbs = ndimage.find_objects(labeled)
    sizes = np.array([[bb.stop - bb.start for bb in line_bb]
                      for line_bb in line_bbs])

    sizes = sizes[:,0]
    mask = (sizes > line_spacing)

    idx = np.flatnonzero(mask)

    for i in idx:
        labels = (labeled == (i+1))
        pixels = indices[labels.T]
        box = [min(pixels[:, 0]), min(pixels[:, 1]), max(pixels[:, 0]), max(pixels[:, 1])]
        rows.append(box)

    return rows

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

    pdf_name, annotations_dir, output_dir, image_dir, gt_dir, thresh = args

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

                if filename.endswith(".csv") or filename.endswith(".pmath"):
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
                        pdf_name,
                        key,
                        annotations_map[pdf_name][key],
                        os.path.join(image_dir, pdf_name, key + '.png'),
                        os.path.join(output_dir, pdf_name, key + '.png'),
                        gt_dir,
                        thresh)

        for math_region in math_regions:
            math_region.insert(0,int(key)-1)
            writer.writerow(math_region)

    math_file.close()


def patch_stitch(filename, annotations_dir, output_dir, image_dir='/home/psm2208/data/GTDB/images/',
                 gt_dir="/home/psm2208/data/GTDB/", thresh=20):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    training_pdf_names_list = []
    training_pdf_names = open(filename, 'r')

    for pdf_name in training_pdf_names:
        pdf_name = pdf_name.strip()

        if pdf_name != '':
            training_pdf_names_list.append((pdf_name, annotations_dir, output_dir, image_dir, gt_dir, thresh))

    training_pdf_names.close()

    for args in training_pdf_names_list:
        stitch_patches(args)

    #pool = Pool(processes=24)
    #pool.map(stitch_patches, training_pdf_names_list)
    #pool.close()
    #pool.join()


if __name__ == '__main__':
    #stitch_patches("TMJ_1990_163_193")
    #stitch_patches("AIF_1970_493_498", "/home/psm2208/code/eval")
    #filename = sys.argv[1] # train_pdf

    #stitch_patches(("Arkiv_1997_185_199", "/home/psm2208/code/eval/Train_Focal_100",
    #             "/home/psm2208/code/eval/Train_Focal_100/out", '/home/psm2208/data/GTDB/images/'))

    #patch_stitch(filename, sys.argv[2], sys.argv[3])
    stride = 0.1

    #stitch_patches(("InvM_1999_163_181", "/home/psm2208/code/eval/Train_Focal_10_25",
    #                "/home/psm2208/code/eval/Train_Focal_10_25/voting_equal_rows",
    #                '/home/psm2208/data/GTDB/images/'))

    thresh = float(sys.argv[1]) # 30
    algorithm = sys.argv[2] # equal
    type = sys.argv[3]  # train_pdf
    dir_to_eval = sys.argv[4] # Test3_Focal_10_25

    home_data = "/home/psm2208/data/GTDB/"
    home_eval = "/home/psm2208/code/eval/"
    home_images = "/home/psm2208/data/GTDB/images/"
    home_anno = "/home/psm2208/data/GTDB/annotations/"

    patch_stitch(home_data + type, home_eval + dir_to_eval,
                 home_eval + dir_to_eval + "/" + algorithm + "_" + str(thresh),
                 home_images, home_anno, thresh)

    # patch_stitch("/home/psm2208/data/GTDB/test_pdf", "/home/psm2208/code/eval/Test_char_Focal_10_25",
    #              "/home/psm2208/code/eval/Test_char_Focal_10_25/voting_equal_" + str(thresh),
    #              '/home/psm2208/data/GTDB/images/', "/home/psm2208/data/GTDB/annotations/", thresh)
