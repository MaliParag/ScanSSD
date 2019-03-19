# Author: Parag Mali

import sys

def split(math_file):

    math_ip = open(math_file, "r")

    # create a map of page to list of math boxes
    map = {}

    for line in math_ip:
        entries = line.strip().split(",")

        # if entry is not in map
        if entries[0] not in map:
            map[entries[0]] = []

        map[entries[0]].append(entries[1:]);

    for key in map:

        boxes = map[key]

        # create processed math file
        math_op = open(str(int(key) + 1) + ".pmath", "w")

        for box in boxes:
            # xmin, xmax, ymin, ymax
            math_op.write(box[0] + "," + box[1] + "," + box[2] + "," + box[3] + "\n")

        math_op.close()

    math_ip.close()

def test():
    split(sys.argv[1])

if __name__ == "__main__":
    test()
