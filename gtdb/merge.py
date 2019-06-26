# Rectangles after projection

import sys
sys.path.extend(['/home/psm2208/code', '/home/psm2208/code'])
import cv2
import os
import csv
import numpy as np
from multiprocessing import Pool
import shutil
import math
from gtdb import feature_extractor
import pickle
from sklearn.ensemble import RandomForestClassifier
from gtdb import fit_box

clf = pickle.load(open('/home/psm2208/code/gtdb/random_forest.pkl', 'rb'))

def intersects(first, other):
    return not (first[2] < other[0] or
                first[0] > other[2] or
                first[1] > other[3] or
                first[3] < other[1])


def merge(args):

    try:
        output_dir, image_file, pdf_name, page_num, det_page_math = args

        image = cv2.imread(image_file)
        im_bw = fit_box.convert_to_binary(image)

        merge_graph = {}

        for i, _ in enumerate(det_page_math):
            merge_graph[i] = set()

        final_math = []
        for math in det_page_math:
            box = fit_box.adjust_box(im_bw, math)
            final_math.append(box)

        det_page_math = final_math

        # find the closest det for each detection
        for i, det_math1 in enumerate(det_page_math):

            x1 = det_math1[0] + (det_math1[2] - det_math1[0] / 2)
            y1 = det_math1[1] + (det_math1[3] - det_math1[1] / 2)

            min = float('inf')
            min_idx = -1

            for j, det_math in enumerate(det_page_math):
                if i != j:
                    x2 = det_math[0] + (det_math[2] - det_math[0] / 2)
                    y2 = det_math[1] + (det_math[3] - det_math[1] / 2)

                    c_dist = (y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1)

                    if c_dist < min:
                        min = c_dist
                        min_idx = j


            features = feature_extractor.extract_features_test(det_page_math[i], det_page_math[min_idx])
            prediction = clf.predict([features])

            if prediction[0] == 1:
                merge_graph[i].add(min_idx)
                merge_graph[min_idx].add(j)

        math_regions_graph = group_math(merge_graph)
        math_regions = create_bb(math_regions_graph, det_page_math)

        output_file = os.path.join(output_dir, pdf_name + ".csv")
        writer = csv.writer(open(output_file, "a"), delimiter=",")

        for math_region in math_regions:
            math_region.insert(0, int(page_num))
            writer.writerow(math_region)

        print("Saved ", output_file, " > ", page_num)

    except:
        print("Exception while processing ", pdf_name, " ", page_num, " ", sys.exc_info())

def create_bb(math_regions_chars, char_info):

    math_regions = []

    for region in math_regions_chars:
        box = []

        for char_id in region:

            if len(box) == 0:
                box = [char_info[char_id][0], char_info[char_id][1],
                       char_info[char_id][2], char_info[char_id][3]]
            else:
                box[0] = min(char_info[char_id][0], box[0])  # left
                box[1] = min(char_info[char_id][1], box[1])  # top
                box[2] = max(char_info[char_id][2], box[2])  # left + width
                box[3] = max(char_info[char_id][3], box[3])  # top + height

        math_regions.append(box)

    return math_regions


def group_math(char_map):

    visited = set()
    regions = []

    for key in char_map:
        if key not in visited:
            region = dfs(char_map, key)
            regions.append(region)

            for k in region:
                visited.add(k)

    return regions


def dfs(graph, start):
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited


def merge_segmentation(filename, image_dir, det_math_dir, output_dir):

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pages_list = []
    pdf_names = open(filename, 'r')

    for pdf_name in pdf_names:
        print('Processing-1', pdf_name)
        pdf_name = pdf_name.strip()

        if pdf_name != '':
            det_math_file = os.path.join(det_math_dir, pdf_name + ".csv")
            det_math_regions = np.genfromtxt(det_math_file, delimiter=',', dtype=int)

            pages = np.unique(det_math_regions[:, 0])

            for page_num in pages:

                det_page_math = det_math_regions[np.where(det_math_regions[:, 0] == page_num)]
                det_page_math = det_page_math[:, 1:]

                pages_list.append([output_dir, os.path.join(image_dir,
                                                        pdf_name,
                                                        page_num + ".png"),
                                           pdf_name, page_num, det_page_math])

    pdf_names.close()

    pool = Pool(processes=1)
    pool.map(merge, pages_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    home_data = "/home/psm2208/data/GTDB/"
    home_eval = "/home/psm2208/code/eval/"
    home_images = "/home/psm2208/data/GTDB/images/"
    home_anno = "/home/psm2208/data/GTDB/annotations/"
    home_char = "/home/psm2208/data/GTDB/char_annotations/"

    output_dir = "/home/psm2208/code/eval/Test3_Focal_10_25/equal_30.0_segmented"

    det_math = "/home/psm2208/code/eval/Test3_Focal_10_25/equal_30.0"

    type = sys.argv[1]

    #filename, gt_math_dir, det_math_dir, output_dir
    merge_segmentation(home_data + type, home_images, det_math, output_dir)
