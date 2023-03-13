import csv
import cv2
import os
import sys
import argparse

"""
    This module annotates math regions and character information on input document images.
    Usage: "python3 visualize_annotations.py --img_dir image_dir --math_dir math_dir --out_dir output_dir --char_dir char_dir"
"""

class Doc():
    '''
        Object to store character and math region information.
    '''
    def __init__(self):
        self.sheet_math_map = {}
        self.sheet_char_map = {}

    def add_math_region(self, row):
        '''
        add math bouding box
        :param row:
        :return: None
        '''
        if int(row[0]) not in self.sheet_math_map:
            self.sheet_math_map[int(row[0])] = []

        self.sheet_math_map[int(row[0])].append((float(row[1]), float(row[2]), float(row[3]), float(row[4])))

    def add_char_region(self, row):
        '''
        add math bouding box
        :param row:
        :return: None
        '''
        if int(row[0]) not in self.sheet_char_map:
            self.sheet_char_map[int(row[0])] = []

        self.sheet_char_map[int(row[0])].append((float(row[2]), float(row[3]), float(row[4]), float(row[5])))

    def read_file(self, filename, is_math = False):
        '''
        Read .math file and parse contents into Doc() object
        :param filename: ".math" annotations filepath
        :param doc: Doc object to store math regions (BBox)
        :return: None
        '''
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if is_math:
                    self.add_math_region(row)
                else:
                    self.add_char_region(row)
                line_count += 1
            print('Processed %d lines.'%line_count)


def draw_rectangle_cv(img_file, math_bboxes, output_path, char_bboxes):
    '''
    Draw math bounding boxes on image file and save resulting image in output path
    :param img_file: input document image filepath
    :param math_bboxes: List of math bounding boxes
    :param output_path: output directory to store annotated document images
    :return: None
    '''
    img = cv2.imread(img_file)
    if math_bboxes:
        for math_bbox in math_bboxes:
            math_left = math_bbox[0]
            math_top = math_bbox[1]
            math_right = math_bbox[2]
            math_bottom = math_bbox[3]
            cv2.rectangle(img, (int(math_left), int(math_top)), (int(math_right), int(math_bottom)), (255, 0, 0), 3)

    if char_bboxes:
        for char_bbox in char_bboxes:
            char_left = char_bbox[0]
            char_top = char_bbox[1]
            char_right = char_bbox[2]
            char_bottom = char_bbox[3]
            cv2.rectangle(img, (int(char_left), int(char_top)), (int(char_right), int(char_bottom)), (0, 255, 0), 3)

    cv2.imwrite(os.path.join(output_path, img_file.split(os.sep)[-1]), img)


def annotate_each_file(img_dir, math_file, char_file, out_dir):
    '''
    Annotate
    :param img_dir: image directory containing all images of a single PDF file
    :param math_file: .math file containing math regions bounding boxes info corresponding to PDF file
    :param out_dir: output path to save annotated document images
    :return:
    '''
    doc = Doc()
    if math_file:
        doc.read_file(math_file, True)
    if char_file:
        doc.read_file(char_file)
    img_files = {}
    for dirName, subdirList, fileList in os.walk(img_dir):
        for img_filename in fileList:
            if img_filename.endswith(".png"):
                img_files[int(img_filename.split(".png")[0]) - 1] = img_filename
        break
    img_name = img_dir.split(os.sep)[-1]
    output_path = os.path.join(out_dir, img_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    for i in sorted(img_files.keys()):
        # print(char_file, math_file)
        if not char_file and math_file:
            if i not in doc.sheet_math_map.keys():
                img = cv2.imread(os.path.join(img_dir, img_files[i]))
                cv2.imwrite(os.path.join(output_path, img_files[i]), img)
                continue
            draw_rectangle_cv(os.path.join(img_dir, img_files[i]), doc.sheet_math_map[i], output_path, None)
        elif char_file and not math_file:
            if i not in doc.sheet_char_map.keys():
                img = cv2.imread(os.path.join(img_dir, img_files[i]))
                cv2.imwrite(os.path.join(output_path, img_files[i]), img)
                continue
            draw_rectangle_cv(os.path.join(img_dir, img_files[i]), None, output_path, doc.sheet_char_map[i])
        elif char_file and math_file:
            if (i not in doc.sheet_math_map.keys()) and (i not in doc.sheet_char_map.keys()):
                img = cv2.imread(os.path.join(img_dir, img_files[i]))
                cv2.imwrite(os.path.join(output_path, img_files[i]), img)
                continue
            elif (i in doc.sheet_math_map.keys()) and (i in doc.sheet_char_map.keys()):
                draw_rectangle_cv(os.path.join(img_dir, img_files[i]), doc.sheet_math_map[i], output_path, doc.sheet_char_map[i])
            elif (i in doc.sheet_math_map.keys()) and (i not in doc.sheet_char_map.keys()):
                draw_rectangle_cv(os.path.join(img_dir, img_files[i]), doc.sheet_math_map[i], output_path, None)
            elif (i not in doc.sheet_math_map.keys()) and (i in doc.sheet_char_map.keys()):
                draw_rectangle_cv(os.path.join(img_dir, img_files[i]), None, output_path, doc.sheet_char_map[i])



def visualize(img_dir,out_dir,math_dir,char_dir=None):


    '''
        store image directory paths
    '''
    img_sub_dirs = {}
    for dirName, subdirList, fileList in os.walk(img_dir):
        for subdir in subdirList:
            if subdir not in img_sub_dirs:
                img_sub_dirs[subdir] = os.path.join(img_dir,subdir)
        break

    '''
        store math files and char files paths
    '''
    math_files = {}
    if math_dir:
        for dirName, subdirList, fileList in os.walk(math_dir):
            for filename in fileList:
                if filename.endswith(".math"):
                    math_files[filename.split(".math")[0]] = os.path.join(math_dir, filename)
            break

    char_files = {}
    if char_dir:
        for dirName, subdirList, fileList in os.walk(char_dir):
            for filename in fileList:
                if filename.endswith(".char"):
                    char_files[filename.split(".char")[0]] = os.path.join(char_dir, filename)
            break


    for key in img_sub_dirs.keys():
        #print(img_sub_dirs[key])
        # print(math_files[key])
        # print(char_files[key])
        if (key in char_files) and (key in math_files):
            annotate_each_file(img_sub_dirs[key], math_files[key], char_files[key], out_dir)
        elif (key in char_files) and (key not in math_files) :
            annotate_each_file(img_sub_dirs[key], None, char_files[key], out_dir)
        elif (key not in char_files) and (key in math_files):
            annotate_each_file(img_sub_dirs[key], math_files[key], None, out_dir)
        else:
            print(key + " not present in annotations")

    return list(math_files.values()),list(img_sub_dirs.values())

