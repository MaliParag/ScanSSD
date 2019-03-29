import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_boxes(im, boxes, path):

    #im = np.array(Image.open('/home/psm2208/data/VOCdevkit/VOC2007/JPEGImages/004602.jpg'), dtype=np.uint8)

    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    #rect = patches.Rectangle((56,138),427,272-138,linewidth=1,edgecolor='r',facecolor='none')

    for box in boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3] - box[1],
                                 linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    #plt.show()
    plt.savefig(path, dpi=600)
    plt.close()


if __name__ == "__main__":
    draw_boxes()