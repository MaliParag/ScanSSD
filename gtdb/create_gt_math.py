# Author: Parag Mali
# This script reads ground truth to find the Symbol Layout Tree (SLT) bounding boxes

import sys
sys.path.extend(['/home/psm2208/code', '/home/psm2208/code'])
import cv2
import os
import csv
import numpy as np
from multiprocessing import Pool
import shutil

def find_math(args):

    try:
        pdf_name, image_file, char_file, page_num, output_file = args

        char_info = {}
        char_map = {}

        image = cv2.imread(image_file)

        with open(char_file) as csvfile:
            char_reader = csv.reader(csvfile, delimiter=',')
            for row in char_reader:
                char_info[row[1]] = row[2:]

                if row[-3] != 'NONE':
                    if row[1] not in char_map:
                        char_map[row[1]] = set()

                    char_map[row[1]].add(row[-2])

                    if row[-2] not in char_map:
                        char_map[row[-2]] = set()

                    char_map[row[-2]].add(row[1])

                elif row[-4] == 'MATH_SYMBOL':
                    if row[1] not in char_map:
                        char_map[row[1]] = set()

        math_regions_chars = group_math(char_map)
        math_regions = create_bb(math_regions_chars, char_info)

        multi_char_math = set({x for v in math_regions_chars for x in v})

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        writer = csv.writer(open(output_file,"a"), delimiter=",")


        # with open(char_file) as csvfile:
        #     char_reader = csv.reader(csvfile, delimiter=',')
        #
        #     for row in char_reader:
        #         if row[-1] in math_ocr and row[0] not in multi_char_math:
        #             math_regions.append([row[2],row[3],row[4],row[5]])

        #math_regions = adjust_all(image, math_regions)

        for math_region in math_regions:
            math_region.insert(0, int(page_num) - 1)
            writer.writerow(math_region)

        print("Saved ", output_file, " > ", page_num, " math ->", len(math_regions))
    except:
        print("Exception while processing ", pdf_name, " ", page_num, " ", sys.exc_info())


def create_bb(math_regions_chars, char_info):

    math_regions = []

    for region in math_regions_chars:
        box = []

        count = 0
        for char_id in region:

            if len(box) == 0:
                box = [float(char_info[char_id][0]),float(char_info[char_id][1]),
                       float(char_info[char_id][2]), float(char_info[char_id][3])]
            else:
                box[0] = min(float(char_info[char_id][0]), box[0])  # left
                box[1] = min(float(char_info[char_id][1]), box[1])  # top
                box[2] = max(float(char_info[char_id][2]), box[2])  # left + width
                box[3] = max(float(char_info[char_id][3]), box[3])  # top + height

            count = count + 1

        box.append(count)
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


def adjust_box(args):
    im_bw, box = args
    box = contract(im_bw, box)
    box = expand(im_bw, box)
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

def adjust_all(image, boxes):

    im_bw = convert_to_binary(image)
    adjusted = []

    for box in boxes:
        box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        box = adjust_box((im_bw, box))
        adjusted.append(box)

    return adjusted

def adjust_box(args):
    im_bw, box = args
    box = contract(im_bw, box)
    box = expand(im_bw, box)
    return box

def create_gt_math(filename, image_dir, char_dir, output_dir="/home/psm2208/data/GTDB/annotationsV2/"):

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pages_list = []
    pdf_names = open(filename, 'r')

    for pdf_name in pdf_names:
        pdf_name = pdf_name.strip()

        if pdf_name != '':

            for root, dirs, files in os.walk(os.path.join(char_dir, pdf_name)):
                for name in files:
                    if name.endswith(".pchar"):

                        page_num = os.path.splitext(name)[0]

                        pages_list.append((pdf_name,
                                           os.path.join(image_dir,
                                                        pdf_name,
                                                        page_num + ".png"),
                                           os.path.join(root, name),
                                           int(page_num),
                                           os.path.join(output_dir,
                                                        pdf_name + ".csv")))
                                                        #page_num + ".pmath")))
    pdf_names.close()

    pool = Pool(processes=32)
    pool.map(find_math, pages_list)
    pool.close()
    pool.join()

if __name__ == "__main__":
    home_data = "/home/psm2208/data/GTDB/"
    home_eval = "/home/psm2208/code/eval/"
    home_images = "/home/psm2208/data/GTDB/images/"
    home_anno = "/home/psm2208/data/GTDB/annotations/"
    home_char = "/home/psm2208/data/GTDB/char_annotations/"
    output_dir = "/home/psm2208/code/eval/tt_samsung_train/"

    type = sys.argv[1]

    create_gt_math(home_data + type, home_images, home_char, output_dir)
