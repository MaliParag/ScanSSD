import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def plot(impath, boxes):

    im = cv2.imread(impath)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    # DET
    box = boxes[0]
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3] - box[1],
                             linewidth=1, edgecolor='b', facecolor='none')
    #Add the patch to the Axes
    ax.add_patch(rect)


    # GT
    box = boxes[1]
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3] - box[1],
                             linewidth=1, edgecolor='r', facecolor='none')
    #Add the patch to the Axes
    ax.add_patch(rect)


    plt.title("GT Test")
    plt.show()
    #plt.savefig(path, dpi=600)
    plt.close()

if __name__ == "__main__":

    boxes = []

    impath = "/Users/parag/Workspace/ssd.pytorch/test/best/voting_equal_30.0/AnnM_1970_550_569/12.png"
    box1 = [3441, 871, 3626, 956] # DET
    box2 = [3409, 871, 3591, 956] # GT

    boxes.append(box1)
    boxes.append(box2)

    plot(impath, boxes)
