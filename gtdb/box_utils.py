# Author: Parag Mali
# This file contains many common operations on the bounding boxes like finding intersection
from collections import OrderedDict
import gtdb.feature_extractor

def check_inside(rectA, rectB):

    # returns True if A is inside B
    if rectA[0] >= rectB[0] and\
       rectA[1] >= rectB[1] and\
       rectA[2] <= rectB[2] and\
       rectA[3] <= rectB[3]:

       return True

    return False

# check if two rectangles intersect
def intersects(first, other):
    return not (first[2] < other[0] or
                first[0] > other[2] or
                first[1] > other[3] or
                first[3] < other[1])

def find_intersecting_boxes(math_regions, sorted=True):
    # specific for fusion algo
    # returns indices of intersecting math region for each math region
    inter_map = {}

    for i in range(len(math_regions)):
        inter_map[i] = []

    for i in range(len(math_regions)):
        for j in range(len(math_regions[i+1:])):
            if intersects(math_regions[i], math_regions[i+1+j]):
                inter_map[i].append(i+1+j)
                #inter_map[i+1+j].append(i)

    inter_map = OrderedDict(inter_map)

    return inter_map

def merge(box1, box2):

    final_box = [0,0,0,0]
    final_box[0] = min(box1[0], box2[0])  # left
    final_box[1] = min(box1[1], box2[1])  # top
    final_box[2] = max(box1[2], box2[2])  # left + width
    final_box[3] = max(box1[3], box2[3])  # top + height

    return final_box


if __name__ == '__main__':
    box1 = [849.00,3797.00,1403.00,3890.00]
    box2 = [1169.00,3804.00,1392.00,3886.00]

    print(intersects(box1, box2))
    print(gtdb.feature_extractor.inclusion(box1, box2))
    print(gtdb.feature_extractor.inclusion(box2, box1))
    print(gtdb.feature_extractor.iou(box2, box1))
