# Author: Parag Mali
# This file performs postprocessing on the detection results
# so that it perfectly contains the connected components

import numpy as np
import cv2

def convert_to_binary(image):
    # convert image to binary

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    im_bw = np.zeros(gray_image.shape)
    im_bw[gray_image > 127] = 0
    im_bw[gray_image <= 127] = 1

    return im_bw

def adjust_box_p(args):
    im_bw, box = args
    return adjust_box(im_bw, box)

def adjust_box(im_bw, box):
    # expand or contract the bounding box to fit the math expression
    box = [int(np.round(x)) for x in box]
    box = contract(im_bw, box)
    box = expand(im_bw, box)
    return box

def contract(im_bw, box):

    # find first row with one pixel
    rows_with_pixels = np.any(im_bw[box[1]:box[3], box[0]:box[2]], axis=1)
    cols_with_pixels = np.any(im_bw[box[1]:box[3], box[0]:box[2]], axis=0)

    if len(rows_with_pixels==True) == 0 or len(cols_with_pixels==True) == 0:
        box = [0,0,0,0]
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

