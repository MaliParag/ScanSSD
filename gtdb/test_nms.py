import numpy as np
import cv2
from cv2.dnn import NMSBoxes

def perform_nms(math_regions):

    # convert from x1,y1,x2,y2 to x,y,w,h
    math_regions[:, 2] = math_regions[:, 2] - math_regions[:, 0]
    math_regions[:, 3] = math_regions[:, 3] - math_regions[:, 1]

    scores = math_regions[:, 4]
    math_regions = np.delete(math_regions, 4, 1)

    math_regions = math_regions.tolist()
    scores = scores.tolist()

    indices = NMSBoxes(math_regions, scores, 0.2, 0.5)

    indices = [item for sublist in indices for item in sublist]
    math_regions = [math_regions[i] for i in indices]

    math_regions = np.array(math_regions)

    # restore to x1,y1,x2,y2
    math_regions[:, 2] = math_regions[:, 2] + math_regions[:, 0]
    math_regions[:, 3] = math_regions[:, 3] + math_regions[:, 1]

    return math_regions.tolist()

if __name__ == '__main__':
    perform_nms(math_regions)
