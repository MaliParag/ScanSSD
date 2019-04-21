# Author: Parag Mali
# This script divides a big image into smaller images using a sliding window.
# It also divides corresponding bounding box annotations.

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
final_width = 300
final_height = 300
stride = 0.5

n_horizontal = int(intermediate_width / crop_size)  # 4
n_vertical = int(intermediate_height / crop_size)  # 5

# This function generates sub-images
def generate_subimages(pdf_name ='Alford94',
                       math_dir='/home/psm2208/data/GTDB/annotations/',
                       image_dir='/home/psm2208/data/GTDB/images/',
                       output_image_dir='/home/psm2208/data/GTDB/processed_images_50/',
                       output_math_dir='/home/psm2208/data/GTDB/processed_annotations_50/'):

    # find all the images
    image_filelist = [file for file in os.listdir(os.path.join(image_dir, pdf_name)) if file.endswith('.png')]

    # math annotations
    math_filepath = os.path.join(math_dir, pdf_name + ".math")
    math_file_present = os.path.isfile(math_filepath)

    if math_file_present:
        math_file = open(math_filepath, 'r')

    if math_file_present:
        boxes = {}

        for line in math_file:
            box = line.split(",")
            idx = int(box[0]) + 1
            box = box[1:]

            box = list(map(int, box))

            if idx not in boxes:
                boxes[idx] = []

            boxes[idx].append(box)

    for image_filepath in image_filelist:

        #os.path.basename
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

        subimg_id = 1

        # create required dirs
        if not os.path.exists(os.path.join(output_image_dir, pdf_name, str(page_id))):
            os.makedirs(os.path.join(output_image_dir, pdf_name, str(page_id)))

        if not os.path.exists(os.path.join(output_math_dir, pdf_name, str(page_id))):
            os.makedirs(os.path.join(output_math_dir, pdf_name, str(page_id)))

        for i in np.arange(0, n_vertical-1+stride, stride):
            for j in np.arange(0, n_horizontal-1+stride, stride):

                print('Processing sub image : ', subimg_id)

                if math_file_present:
                    out_math_file = os.path.join(output_math_dir, pdf_name, str(page_id), str(subimg_id) + ".pmath")
                    out_math = open(out_math_file, "w")

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
                            out_math.write(str(box[0]) + "," + str(box[1]) + "," + str(box[2]) + "," + str(box[3]) + "\n")

                fig_name = os.path.join(output_image_dir, pdf_name, str(page_id), str(subimg_id) + ".png")
                print("Saving " + fig_name)
                cv2.imwrite(fig_name, cropped_image)
                subimg_id = subimg_id + 1

                if math_file_present:
                    out_math.close()
                    if count == 0:
                        # if no math regions, delete the file
                        os.remove(out_math_file)


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

    training_pdf_names.close()

    math_dir = '/home/psm2208/data/GTDB/annotations/'
    image_dir = '/home/psm2208/data/GTDB/images/'
    output_image_dir = '/home/psm2208/data/GTDB/processed_images_50/'
    output_math_dir = '/home/psm2208/data/GTDB/processed_annotations_50/'

    # create required dirs
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    if not os.path.exists(output_math_dir):
        os.makedirs(output_math_dir)

    if not os.path.exists(os.path.join(output_image_dir, pdf_name)):
        os.makedirs(os.path.join(output_image_dir, pdf_name))

    if not os.path.exists(os.path.join(output_math_dir, pdf_name)):
        os.makedirs(os.path.join(output_math_dir, pdf_name))

    pool = Pool(processes=4)
    pool.map(generate_subimages, training_pdf_names_list)
    pool.close()
    pool.join()

