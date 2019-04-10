import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np

def draw_boxes(im, recognized_boxes, boxes, confs, scale, path):

    # Create figure and axes
    fig,ax = plt.subplots(1)
    scale = scale.cpu().numpy()

    # TODO: Added for debugging. Assuming all dimensions scale factor is same
    boxes = boxes.cpu().numpy() * scale[0]
    confs = confs.cpu().numpy().reshape(-1, 1)
    data = np.append(boxes, confs, 1)

    # sort based on the confs. Confs is column 4
    data = data[data[:,4].argsort()]

    # Display the image
    ax.imshow(im)

    width, height, channels = im.shape
    heatmap = np.zeros([width, height])


    for box in data:
        heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = box[4]

    for box in recognized_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3] - box[1],
                                 linewidth=1, edgecolor='r', facecolor='none')
        #Add the patch to the Axes
        ax.add_patch(rect)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    # Following line makes sure that all the heatmaps are in the scale, 0 to 1
    # So color assigned to different scores are consistent across heatmaps for
    # different images
    heatmap[0:1, 0:1] = 1

    plt.imshow(heatmap, alpha=0.4, cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.show()
    plt.savefig(path, dpi=600)
    plt.close()


if __name__ == "__main__":
    draw_boxes()