from __future__ import print_function
import os
import argparse
import torch.backends.cudnn as cudnn
from data import *
from ssd import build_ssd
from utils import draw_boxes, helpers, save_boxes

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_GTDB_990.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--dataset_root', default=GTDB_ROOT, help='Location of VOC root directory')
parser.add_argument('--type', default="test", help='Evaluate on test or train')
parser.add_argument('--verbose', default=False, type=bool, help='plot output')
parser.add_argument('--suffix', default="_10", type=str, help='suffix of directory of images for testing')
parser.add_argument('--exp_name', default="SSD", help='Name of the experiment. Will be used to generate output')
parser.add_argument('--model_type', default=300, type=int,
                    help='Type of ssd model, ssd300 or ssd512')
parser.add_argument('--use_char_info', default=False, type=bool, help='Whether or not to use char info')
parser.add_argument('--limit', default=-1, type=int, help='limit on number of test examples')
parser.add_argument('--cfg', default="gtdb", type=str,
                    help='Type of network: either gtdb or math_gtdb_512')
parser.add_argument('--kernel', default="3 3", type=int, nargs='+',
                    help='Kernel size for feature layers: 3 3 or 1 5')
parser.add_argument('--padding', default="1 1", type=int, nargs='+',
                    help='Padding for feature layers: 1 1 or 0 2')
parser.add_argument('--neg_mining', default=True, type=bool,
                    help='Whether or not to use hard negative mining with ratio 1:3')

parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")

args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(args, net, gpu_id, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = args.save_folder + 'detection_output.txt'
    if os.path.isfile(filename):
        os.remove(filename)

    num_images = len(testset)
    f = open(filename, "w")

    if args.limit != -1:
        num_images = args.limit

    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i, 'test')
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = x.unsqueeze(0)

        f.write('\nFOR: '+img_id+'\n')
        for box in annotation:
            f.write('label: '+' || '.join(str(b) for b in box)+'\n')
        if args.cuda:
            x = x.to(gpu_id)

        y, debug_boxes, debug_scores = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0

        recognized_boxes = []
        recognized_scores = []

        #[1,2,200,5] -> 1 is number of classes, 200 is top_k, 5 is bounding box with class label,
        for i in range(detections.size(1)):
            j = 0
            while j < detections.size(2) and detections[0, i, j, 0] >= thresh: #TODO it was 0.6
                if pred_num == 0:
                    f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = 'math'
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])

                recognized_boxes.append(coords)
                recognized_scores.append(score.cpu().numpy())

                pred_num += 1
                f.write(str(pred_num)+' label: '+label_name+' score: ' +
                        str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1

        save_boxes(args, recognized_boxes, recognized_scores, img_id)
        if args.verbose:
            draw_boxes(args, img, recognized_boxes, debug_boxes, debug_scores, scale, img_id)

    f.close()

def test_gtdb():

    gpu_id = 0
    if args.cuda:
        gpu_id = helpers.get_freer_gpu()
        torch.cuda.set_device(gpu_id)

    # load net
    num_classes = 2 # +1 background

    net = build_ssd(args, 'test', gtdb, gpu_id, args.model_type, num_classes)  # initialize SSD

    print(net)
    net.to(gpu_id)

    # TODO: should remove map_location argument
    net.load_state_dict(torch.load(args.trained_model, map_location={'cuda:0':'cuda:1'}))
    net.eval()
    print('Finished loading model!')

    #testset = GTDBDetection(args.dataset_root, 'processed_train', None, GTDBAnnotationTransform())
    testset = GTDBDetection(args, None, GTDBAnnotationTransform())

    if args.cuda:
        net = net.to(gpu_id)
        cudnn.benchmark = True

    # evaluation
    test_net(args, net, gpu_id, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)


if __name__ == '__main__':
    #test_voc()
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    #torch.cuda.set_device(1)
    test_gtdb()
