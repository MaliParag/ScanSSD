import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import cv2


def draw_stitched_boxes(im, data, outpath):

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # sort based on the confs. Confs is column 4
    data = data[data[:, 4].argsort()]

    # Display the image
    ax.imshow(im)

    width, height, channels = im.shape
    heatmap = np.zeros([width, height])

    for box in data:
        heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = box[4]

    # Following line makes sure that all the heatmaps are in the scale, 0 to 1
    # So color assigned to different scores are consistent across heatmaps for
    # different images
    heatmap[0:1, 0:1] = 1
    heatmap[0:1, 1:2] = 0

    plt.imshow(heatmap, alpha=0.4, cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.title("Stitching visualization")
    plt.show()
    plt.savefig(outpath, dpi=600)
    plt.close()

def draw_all_boxes(im, data, recognized_boxes, gt_boxes, outpath):

    if len(data) == 0:
        return

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # sort based on the confs. Confs is column 4
    data = data[data[:, 4].argsort()]

    # Display the image
    ax.imshow(im)

    width, height, channels = im.shape
    heatmap = np.zeros([width, height])

    if data is not None:
        for box in data:
            heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = box[4]
            #rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
            #                        linewidth=0.25, edgecolor='m', facecolor='none')
            #Add the patch to the Axes
            #ax.add_patch(rect)

    if recognized_boxes is not None:
        # recognized boxes are green
        for box in recognized_boxes:
             rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                      linewidth=1, edgecolor='g', facecolor='none')
             # Add the patch to the Axes
             ax.add_patch(rect)


    if gt_boxes is not None:
        # ground truth are red
        for box in gt_boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=0.25, edgecolor='b', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)

    # Following line makes sure that all the heatmaps are in the scale, 0 to 1
    # So color assigned to different scores are consistent across heatmaps for
    # different images
    heatmap[0:1, 0:1] = 1
    heatmap[0:1, 1:2] = 0

    plt.imshow(heatmap, alpha=0.4, cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.title("Stitching visualization")
    plt.show()
    plt.savefig(outpath, dpi=600)
    plt.close()


def draw_boxes_cv(image, recognized_boxes, gt_boxes, outpath):

    '''
    :param image:
    :param recognized_boxes:
    :param outpath: save as outpath. Should be complete image path with extension
    :return:
    '''

    #(BGR)
    # detected is green
    for box in recognized_boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)

    # ground truth is blue
    for box in gt_boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)

    cv2.imwrite(outpath, image)


def save_boxes(args, recognized_boxes, recognized_scores, img_id):

    path = os.path.join("eval", args.exp_name, img_id + ".png")
    math_csv_path = os.path.join("eval", args.exp_name, img_id + ".csv")

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    math_output = open(math_csv_path, 'w')

    for i, box in enumerate(recognized_boxes):
        math_output.write(str(box[0]) + ',' + str(box[1]) + ',' + str(box[2]) + ',' +
                          str(box[3]) + ',' + str(recognized_scores[i]) + '\n')

    math_output.close()


def draw_boxes(args, im, recognized_boxes, boxes, confs, scale, img_id):

    path = os.path.join("eval", args.exp_name, img_id + ".png")

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    if args.verbose:
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
                                     linewidth=1, edgecolor='g', facecolor='none')
            #Add the patch to the Axes
            ax.add_patch(rect)

        # Following line makes sure that all the heatmaps are in the scale, 0 to 1
        # So color assigned to different scores are consistent across heatmaps for
        # different images
        heatmap[0:1, 0:1] = 1
        heatmap[0:1, 1:2] = 0

        plt.imshow(heatmap, alpha=0.4, cmap='hot', interpolation='nearest')
        plt.colorbar()

        plt.title(args.exp_name)
        plt.show()
        plt.savefig(path, dpi=600)
        plt.close()


if __name__ == "__main__":
    draw_boxes()