# Author: Parag Mali
# This script generates page level annotations from the PDF level annotations
# provided in the dataset

import sys
import os
from multiprocessing import Pool
import csv
import cv2

def split(args):

    gt_dir, pdf_name, out_dir, ext = args

    file_path = os.path.join(gt_dir, pdf_name + "." + ext)
    img_dir = '/home/psm2208/data/GTDB/images/'

    # create a map of page to list of math boxes
    map = {}

    if ext == "math":

        file_ip = open(file_path, "r")
        for line in file_ip:
            entries = line.strip().split(",")

            # if entry is not in map
            if entries[0] not in map:
                map[entries[0]] = []

            map[entries[0]].append(entries[1:])

        for key in map:

            boxes = map[key]
            key = float(key)
            img_file = os.path.join(img_dir, pdf_name, str(int(key) + 1) + ".png")
            img = cv2.imread(img_file)

            height, width, channels = img.shape

            #width_ratio = 512 / width
            #height_ratio = 512 / height

            width_ratio = 1
            height_ratio = 1

            # create processed math file
            file_op = open(os.path.join(out_dir, pdf_name, str(int(key) + 1)) + ".p" + ext, "w")

            for box in boxes:
                # xmin, ymin, xmax, ymax

                box[0] = float(box[0]) * width_ratio
                box[1] = float(box[1]) * height_ratio
                box[2] = float(box[2]) * width_ratio
                box[3] = float(box[3]) * height_ratio

                file_op.write(','.join(str(e) for e in box) + "\n")

            file_op.close()
            file_ip.close()

    elif ext == "char":
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                # if entry is not in map
                if row[0] not in map:
                    map[row[0]] = []

                map[row[0]].append(row)

        for key in map:

            boxes = map[key]

            with open(os.path.join(out_dir, pdf_name, str(int(key) + 1)) + ".p" + ext, "w") as csvfile:
                writer = csv.writer(csvfile, delimiter=',')

                for box in boxes:
                    writer.writerow(box)

def test():

    filename = sys.argv[1] # file names to be processed
    out_dir = sys.argv[2] # output dir
    gt_dir = sys.argv[3] # gt dir
    ext = sys.argv[4] # file extension

    pdf_names_list = []
    pdf_names = open(filename, 'r')

    for pdf_name in pdf_names:
        pdf_name = pdf_name.strip()

        if not os.path.exists(os.path.join(out_dir, pdf_name)):
            os.mkdir(os.path.join(out_dir, pdf_name))

        if pdf_name != '':
            pdf_names_list.append((gt_dir, pdf_name, out_dir, ext))

    pdf_names.close()

    pool = Pool(processes=32)
    pool.map(split, pdf_names_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    test()
