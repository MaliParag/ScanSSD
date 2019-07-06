# check if two rectangles intersect
def intersects(first, other):
    return not (first[2] < other[0] or
                first[0] > other[2] or
                first[1] > other[3] or
                first[3] < other[1])

def find_intersecting_boxes(math_regions):

    # returns indices of intersecting math region for each math region
    inter_map = {}

    for i in range(len(math_regions)):
        inter_map[i] = set()

    for i in range(len(math_regions)):
        for j in range(len(math_regions[i+1:])):
            if intersects(math_regions[i], math_regions[i+1+j]):
                inter_map[i].add(i+1+j)
                inter_map[i+1+j].add(i)

    return inter_map

def merge(box1, box2):

    final_box = [0,0,0,0]
    final_box[0] = min(box1[0], box2[0])  # left
    final_box[1] = min(box1[1], box2[1])  # top
    final_box[2] = max(box1[2], box2[2])  # left + width
    final_box[3] = max(box1[3], box2[3])  # top + height

    return final_box