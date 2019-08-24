# Author: Parag Mali
# This script divides a big image into smaller images using a sliding window.
# It also divides corresponding bounding box annotations.
# NOTE: This file is no longer needed as sub-images are generated in data-loader

# read the image
import numpy as np
import cv2
import copy
import os
import sys
from multiprocessing import Pool

# Default parameters for thr GTDB dataset
intermediate_width = 4800
intermediate_height = 6000
crop_size = 1200
final_width = 0
final_height = 0
stride = 0.1

n_horizontal = int(intermediate_width / crop_size)  # 4
n_vertical = int(intermediate_height / crop_size)  # 5

math_dir='/home/psm2208/data/GTDB/annotations/'
char_dir='/home/psm2208/data/GTDB/char_annotations/'
image_dir='/home/psm2208/data/GTDB/images/'
output_image_dir='/home/psm2208/data/GTDB/processed_images_/'
output_math_dir='/home/psm2208/data/GTDB/processed_annotations_/'
output_char_dir='/home/psm2208/data/GTDB/processed_char_annotations_/'

# This function generates sub-images
def generate_subimages(pdf_name ='Alford94'):

    # find all the images
    image_filelist = [file for file in os.listdir(os.path.join(image_dir, pdf_name)) if file.endswith('.png')]

    # math annotations
    math_filepath = os.path.join(math_dir, pdf_name + ".math")
    math_file_present = os.path.isfile(math_filepath)

    # char annotations
    char_filepath = os.path.join(char_dir, pdf_name + ".char")
    char_file_present = os.path.isfile(char_filepath)

    if math_file_present:
        math_file = open(math_filepath, 'r')

        boxes = {}

        for line in math_file:
            box = line.split(",")
            idx = int(box[0]) + 1
            box = box[1:]

            box = list(map(float, box))
            box = list(map(int, box))

            if idx not in boxes:
                boxes[idx] = []

            boxes[idx].append(box)

    if char_file_present:
        char_file = open(char_filepath, 'r')

        char_boxes = {}

        for line in char_file:
            char_box = line.split(",")
            idx = int(char_box[0]) + 1
            char_box = char_box[2:]

            #box = list(map(int, box))

            if idx not in char_boxes:
                char_boxes[idx] = []

            char_boxes[idx].append(char_box)


    for image_filepath in image_filelist:

        image = cv2.imread(os.path.join(image_dir, pdf_name, image_filepath))
        basename = os.path.basename(image_filepath)
        page_id = int(os.path.splitext(basename)[0])

        original_width = image.shape[1]
        original_height = image.shape[0]

        intermediate_width_ratio = intermediate_width / original_width
        intermediate_height_ratio = intermediate_height / original_height

        image = cv2.resize(image, (intermediate_width, intermediate_height))

        if math_file_present:
            if page_id in boxes:
                current_boxes = boxes[page_id]
            else:
                current_boxes = []

            # preprocess the boxes
            for box in current_boxes:

                box[0] = box[0] * intermediate_width_ratio
                box[1] = box[1] * intermediate_height_ratio
                box[2] = box[2] * intermediate_width_ratio
                box[3] = box[3] * intermediate_height_ratio

        if char_file_present:
            if page_id in char_boxes:
                current_char_boxes = char_boxes[page_id]
            else:
                current_char_boxes = []

            # preprocess the boxes
            for box in current_char_boxes:

                box[0] = float(box[0]) * intermediate_width_ratio
                box[1] = float(box[1]) * intermediate_height_ratio
                box[2] = float(box[2]) * intermediate_width_ratio
                box[3] = float(box[3]) * intermediate_height_ratio


        subimg_id = 1

        # create required dirs
        if not os.path.exists(os.path.join(output_image_dir, pdf_name, str(page_id))):
            os.makedirs(os.path.join(output_image_dir, pdf_name, str(page_id)))

        if not os.path.exists(os.path.join(output_math_dir, pdf_name, str(page_id))):
            os.makedirs(os.path.join(output_math_dir, pdf_name, str(page_id)))

        if not os.path.exists(os.path.join(output_char_dir, pdf_name, str(page_id))):
            os.makedirs(os.path.join(output_char_dir, pdf_name, str(page_id)))

        for i in np.arange(0, n_vertical-1+stride, stride):
            for j in np.arange(0, n_horizontal-1+stride, stride):

                print('Processing sub image : ', subimg_id)

                if math_file_present:
                     out_math_file = os.path.join(output_math_dir, pdf_name, str(page_id), str(subimg_id) + ".pmath")
                     out_math = open(out_math_file, "w")

                if char_file_present:
                    out_char_file = os.path.join(output_char_dir, pdf_name, str(page_id), str(subimg_id) + ".pchar")
                    out_char = open(out_char_file, "w")

                x_l = int(np.round(crop_size * i))
                x_h = int(np.round(crop_size * (i + 1)))

                y_l = int(np.round(crop_size * j))
                y_h = int(np.round(crop_size * (j + 1)))

                cropped_image = image[x_l: x_h, y_l: y_h, :]
                cropped_image = cv2.resize(cropped_image, (final_width, final_height))

                # left, top, right, bottom
                image_box = [y_l, x_l, y_h, x_h]

                # find scaling factors
                final_width_ratio = final_width / (y_h - y_l)
                final_height_ratio = final_height / (x_h - x_l)

                count = 0

                if math_file_present:
                    if page_id in boxes:
                        current_page_boxes = copy.deepcopy(boxes[page_id])
                    else:
                        current_page_boxes = []

                    # if math intersects only consider the region which
                    # is part of the current bounding box
                    for box in current_page_boxes:
                        if intersects(image_box, box):
                            #print('intersects ', box)

                            # left, top, right, bottom
                            # y increases downwards

                            #crop the boxes to fit into image region
                            box[0] = max(y_l, box[0])
                            box[1] = max(x_l, box[1])
                            box[2] = min(y_h, box[2])
                            box[3] = min(x_h, box[3])

                            # Translate to origin
                            box[0] = box[0] - y_l
                            box[2] = box[2] - y_l

                            box[1] = box[1] - x_l
                            box[3] = box[3] - x_l

                            # scaling
                            box[2] = int(np.round(box[2] * final_width_ratio))
                            box[0] = int(np.round(box[0] * final_width_ratio))

                            box[3] = int(np.round(box[3] * final_height_ratio))
                            box[1] = int(np.round(box[1] * final_height_ratio))

                            count = count + 1
                            out_math.write(','.join(str(x) for x in box))
                            out_math.write("\n")
                            #cv2.rectangle(cropped_image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)

                if char_file_present:
                    if page_id in char_boxes:
                        current_page_char_boxes = copy.deepcopy(char_boxes[page_id])
                    else:
                        current_page_char_boxes = []

                    # if math intersects only consider the region which
                    # is part of the current bounding box
                    for box in current_page_char_boxes:
                        if intersects(image_box, box):
                            #print('intersects ', box)

                            # left, top, right, bottom
                            # y increases downwards

                            #crop the boxes to fit into image region
                            box[0] = max(y_l, box[0])
                            box[1] = max(x_l, box[1])
                            box[2] = min(y_h, box[2])
                            box[3] = min(x_h, box[3])

                            # Translate to origin
                            box[0] = box[0] - y_l
                            box[2] = box[2] - y_l

                            box[1] = box[1] - x_l
                            box[3] = box[3] - x_l

                            # scaling
                            box[2] = int(np.round(box[2] * final_width_ratio))
                            box[0] = int(np.round(box[0] * final_width_ratio))

                            box[3] = int(np.round(box[3] * final_height_ratio))
                            box[1] = int(np.round(box[1] * final_height_ratio))

                            out_char.write(','.join(str(x) for x in box))
                            #out_char.write("\n")

                fig_name = os.path.join(output_image_dir, pdf_name, str(page_id), str(subimg_id) + ".png")
                print("Saving " + fig_name)
                cv2.imwrite(fig_name, cropped_image)
                subimg_id = subimg_id + 1

                if math_file_present:

                    out_math.close()
                    out_char.close()
                    if count == 0:
                        # if no math regions, delete the file
                        os.remove(out_math_file)
                        os.remove(out_char_file)


# check if two rectangles intersect
def intersects(first, other):
    return not (first[2] < other[0] or
                first[0] > other[2] or
                first[1] > other[3] or
                first[3] < other[1])

if __name__ == '__main__':

    training_pdf_names = open(sys.argv[1], 'r')
    stride = float(sys.argv[2]) # 0.1

    training_pdf_names_list = []

    # for each training image pdf file
    for pdf_name in training_pdf_names:
        pdf_name = pdf_name.strip()
        if pdf_name != '':
            training_pdf_names_list.append(pdf_name)

            if not os.path.exists(os.path.join(output_image_dir, pdf_name)):
                os.makedirs(os.path.join(output_image_dir, pdf_name))

            if not os.path.exists(os.path.join(output_math_dir, pdf_name)):
                os.makedirs(os.path.join(output_math_dir, pdf_name))

            if not os.path.exists(os.path.join(output_char_dir, pdf_name)):
                os.makedirs(os.path.join(output_char_dir, pdf_name))

    training_pdf_names.close()

    suffix = sys.argv[3] #str(int(100 * stride))

    final_height = int(suffix)
    final_width = int(suffix)

    char_dir = '/home/psm2208/data/GTDB/char_annotations/'
    math_dir = '/home/psm2208/code/eval/relations_adjust/'
    #math_dir = '/home/psm2208/data/GTDB/annotations/'
    image_dir = '/home/psm2208/data/GTDB/images/'

    output_image_dir = '/home/psm2208/data/GTDB/processed_images_' + suffix
    output_math_dir = '/home/psm2208/data/GTDB/processed_annotations_' + suffix
    output_char_dir = '/home/psm2208/data/GTDB/processed_char_annotations_' + suffix

    # create required dirs
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    if not os.path.exists(output_math_dir):
        os.makedirs(output_math_dir)

    if not os.path.exists(output_char_dir):
        os.makedirs(output_char_dir)

    pool = Pool(processes=32)
    pool.map(generate_subimages, training_pdf_names_list)
    pool.close()
    pool.join()
