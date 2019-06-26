import math

def extract_features_test(box1, box2):

    features = [area(box1), height(box1), width(box1),
                area(box2), height(box2), width(box2),
                center_dist(box1, box2),
                horizontal_dist_center(box1, box2),
                vertical_dist_center(box1, box2),
                vertical_dist_bb(box1, box2)] # bottom of box to top of box, min

    return features

def extract_features(box1, box2, label=1):

    features = [area(box1), height(box1), width(box1),
                area(box2), height(box2), width(box2),
                center_dist(box1, box2),
                horizontal_dist_center(box1, box2),
                vertical_dist_center(box1, box2),
                vertical_dist_bb(box1, box2), label] # bottom of box to top of box, min

    return features

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
