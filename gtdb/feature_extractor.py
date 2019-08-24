# Author: Parag Mali
# This file contains functions to extract a set of features from two given
# bounding boxes (eg. Geometric features)

import math

def extract_features(box1, box2, label=1, test=False):

    features = [area(box1)/area(box2),
                height(box1)/height(box2),
                width(box1)/width(box2),
                center_dist(box1, box2),
                horizontal_dist_center(box1, box2),
                vertical_dist_center(box1, box2),
                vertical_dist_bb(box1, box2),
                horizontal_indentation(box1, box2),
                aspect_ratio(box1),
                aspect_ratio(box2)] # bottom of box to top of box, min

    if not test:
        features.append(label)

    return features


def intersection(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    return interArea

def union(boxA, boxB):
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    return float(boxAArea + boxBArea - intersection(boxA, boxB))


def iou(boxA, boxB):
    return intersection(boxA, boxB) / union(boxA, boxB)

def inclusion(box1, box2):
    return intersection(box1, box2) / area(box1)

def aspect_ratio(box):
    return width(box)/height(box)

def horizontal_indentation(box1, box2):
    return abs(box1[0]-box2[0])

def width(box):
    return box[2] - box[0]

def height(box):
    return box[3] - box[1]

def area(box):
    return width(box) * height(box)

def center_dist(box1, box2):
    x1 = box1[0] + (width(box1) / 2)
    y1 = box1[1] + (height(box1) / 2)
    x2 = box2[0] + (width(box2) / 2)
    y2 = box2[1] + (height(box2) / 2)

    return math.sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1))

def horizontal_dist_center(box1, box2):
    x1 = box1[0] + (width(box1) / 2)
    x2 = box2[0] + (width(box2) / 2)

    return abs(x2-x1)

def vertical_dist_center(box1, box2):
    y1 = box1[1] + (height(box1) / 2)
    y2 = box2[1] + (height(box2) / 2)

    return abs(y2-y1)

def vertical_dist_bb(box1, box2):
    return min(abs(box1[3]-box2[1]), abs(box2[3]-box1[1]))
