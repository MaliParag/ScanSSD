# Rectangles after projection

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
import copy
import shutil

def check_collision(rectA,  rectB):

    # returns True if A is inside B
    #left, top, right, bottom
    #If any of the sides from A are outside of B
    if rectA[3] < rectB[1]: # if bottom of rectA is less than top of rectB
        return False
    if rectA[1] > rectB[3]: # if top of rectA is greater than bottom of rectB
        return False
    if rectA[2] < rectB[0]: # if right of rectA is less than left of rectB
        return False
    if rectA[0] > rectB[2]: # if left of rectangleA is greater than right of rectB
        return False

    #If none of the sides from A are outside B
    return True


def remove(args):

    try:
        output_dir, pdf_name, page_num, page_math = args

        valid = [True] * page_math.shape[0]

        for i, m1 in enumerate(page_math):
            for j, m2 in enumerate(page_math[i+1:]):
                if check_collision(m1, m2):
                    valid[i] = False
                    break

        final_math = page_math[valid]

        math_file = open(os.path.join(output_dir, pdf_name + ".csv"), 'a')
        writer = csv.writer(math_file, delimiter=",")

        for math_region in final_math:
            math_region = math_region.tolist()
            math_region.insert(0, page_num)
            writer.writerow(math_region)

        print("Saved ", os.path.join(output_dir, pdf_name + ".csv"), " > ", page_num)
        print('Before ', len(page_math), '--> after ', len(final_math))

    except:
        print("Exception while processing ", pdf_name, " ", page_num, " ", sys.exc_info()[0])


def remove_rect(filename, math_dir, output_dir):

    shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pages_list = []
    pdf_names = open(filename, 'r')

    for pdf_name in pdf_names:
        print('Processing-1', pdf_name)
        pdf_name = pdf_name.strip()

        if pdf_name != '':
            math_file = os.path.join(math_dir, pdf_name + ".csv")
            math_regions = np.genfromtxt(math_file, delimiter=',', dtype=int)

            pages = np.unique(math_regions[:, 0])

            for page_num in pages:

                page_math = math_regions[np.where(math_regions[:,0]==page_num)]
                page_math = page_math[:,1:]
                pages_list.append([output_dir, pdf_name, page_num, page_math])

    pdf_names.close()

    pool = Pool(processes=24)
    pool.map(remove, pages_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    home_data = "/home/psm2208/data/GTDB/"
    home_eval = "/home/psm2208/code/eval/"
    home_images = "/home/psm2208/data/GTDB/images/"
    home_anno = "/home/psm2208/data/GTDB/annotations/"

    math_dir = "/home/psm2208/data/GTDB/relations_adjust_csv/"
    output_dir = "/home/psm2208/data/GTDB/relations_adjust_csv_removed/"

    type = sys.argv[1]

    remove_rect(home_data + type, math_dir, output_dir)
