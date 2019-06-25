import sys
sys.path.extend(['/home/psm2208/code', '/home/psm2208/code'])
import cv2
import os
import csv
import numpy as np
import utils.visualize as visualize
from multiprocessing import Pool
from scipy.ndimage.measurements import label
import scipy.ndimage as ndimage
import copy

final_width = 512
final_height = 512

def draw_bb(args):

    try:
        pdf_name, image_path, ppm_file, char_file, page_num, output_file = args

        image = cv2.imread(image_path)

        math_regions = np.genfromtxt(ppm_file, delimiter=',')

        # if there is only one entry convert it to correct form required
        if len(math_regions.shape) == 1:
            math_regions = math_regions.reshape(1, -1)

        height, width, channels = image.shape
        final_image = np.zeros([512,512,3])

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image,(final_width,final_height),interpolation=cv2.INTER_AREA)

        final_image[:, :, 0] = gray_image # Blue

        width_ratio = final_width / width
        height_ratio = final_height / height

        boundary_scores = np.zeros([final_width, final_height])
        boundary_confs = np.zeros([final_width, final_height])

        # sort
        if len(math_regions) > 4:
            math_regions = math_regions[math_regions[:, 4].argsort()]

            for box in math_regions:
                box = [box[0]*width_ratio, box[1]*height_ratio, box[2]*width_ratio, box[3]*height_ratio, box[4]]
                boundary_scores[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = \
                    boundary_scores[int(box[1]):int(box[3]), int(box[0]):int(box[2])] + 1
                boundary_confs[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = box[4] * 2.55 #(255 / 100)


        final_image[:, :, 1] = boundary_scores # Green
        final_image[:, :, 2] = boundary_confs # Red

        cv2.imwrite(output_file, final_image)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print('Saved ', output_file)

    except:
        print('Error occurred while processing ', output_file,  sys.exc_info())

def draw_math_bb(filename, ppm_dir, image_dir, char_dir, output_dir):


    pages_list = []
    pdf_names = open(filename, 'r')

    for pdf_name in pdf_names:
        pdf_name = pdf_name.strip()

        if pdf_name != '':

            for root, dirs, files in os.walk(os.path.join(ppm_dir, pdf_name)):
                for name in files:
                    if name.endswith(".ppm"):

                        page_num = os.path.splitext(name)[0]

                        pages_list.append((pdf_name,
                                           os.path.join(image_dir,
                                                        pdf_name,
                                                        page_num + ".png"),
                                           os.path.join(ppm_dir,
                                                        pdf_name,
                                                        page_num + ".ppm"),
                                           os.path.join(root, name),
                                           int(page_num),
                                           os.path.join(output_dir,
                                                        pdf_name,
                                                        page_num + ".png")))

    pdf_names.close()


    pool = Pool(processes=24)
    pool.map(draw_bb, pages_list)
    pool.close()
    pool.join()

if __name__ == "__main__":
    home_data = "/home/psm2208/data/GTDB/"
    home_eval = "/home/psm2208/code/eval/"
    home_images = "/home/psm2208/data/GTDB/images/"
    home_anno = "/home/psm2208/data/GTDB/annotations/"
    home_char = "/home/psm2208/data/GTDB/char_annotations/"
    output_dir = "/home/psm2208/data/GTDB/processed_images_stage_2/"
    ppm_dir = "/home/psm2208/code/eval/Test3_Focal_10_25/" # post-processed math after detection

    type = sys.argv[1]

    draw_math_bb(home_data + type, ppm_dir, home_images, home_char, output_dir)
