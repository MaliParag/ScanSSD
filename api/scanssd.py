'''
Arquivo com as funções necessárias para Teste do ScanSSD
'''
from __future__ import print_function
import os,sys
from PIL import Image
import torch.backends.cudnn as cudnn
from ssd import build_ssd
from utils import draw_boxes, helpers, save_boxes
import logging
import time
import datetime
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from data import *
import shutil
import torch.nn as nn
from api.utils_scanssd.arguments import Arguments
from api.utils_scanssd.convert_pdf_to_image import create_images_from_pdfs
from gtdb.stitch_pages import stitch
from api.utils_scanssd.visualize_annotations import visualize
from api.utils_scanssd.crop_resize_image import crop_resize

#from pix2tex import cli as pix2tex



def test_net_batch(args, net, gpu_id, dataset, transform, thresh):
    '''
    ScanSSD Test
    '''    
    num_images = len(dataset)

    if args.limit != -1:
        num_images = args.limit

    data_loader = DataLoader(dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=False, collate_fn=detection_collate,
                              pin_memory=True)

    total = len(dataset)

    logging.debug('Test dataset size is {}'.format(total))

    done = 0

    for batch_idx, (images, targets, metadata) in enumerate(data_loader):

        done = done + len(images)
        logging.debug('processing {}/{}'.format(done, total))

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        else:
            images = Variable(images,requires_grad=False)
            targets = [Variable(ann, requires_grad=False) for ann in targets]

        y, debug_boxes, debug_scores = net(images)  # forward pass
        detections = y.data

        k = 0
        for img, meta in zip(images, metadata):

            img_id = meta[0]
            x_l = meta[1]
            y_l = meta[2]

            img = img.permute(1,2,0)
            # scale each detection back up to the image
            scale = torch.Tensor([img.shape[1], img.shape[0],
                                  img.shape[1], img.shape[0]])

            recognized_boxes = []
            recognized_scores = []

            # [1,2,200,5]
            # we only care about math class
            # hence select detections[image_id, class, detection_id, detection_score]
            # class=1 for math
            i = 1
            j = 0

            while j < detections.size(2) and detections[k, i, j, 0] >= thresh:  # TODO it was 0.6

                score = detections[k, i, j, 0]
                pt = (detections[k, i, j, 1:] * args.window).cpu().numpy()
                coords = (pt[0] + x_l, pt[1] + y_l, pt[2] + x_l, pt[3] + y_l)
                #coords = (pt[0], pt[1], pt[2], pt[3])
                recognized_boxes.append(coords)
                recognized_scores.append(score.cpu().numpy())

                j += 1

            save_boxes(args, recognized_boxes, recognized_scores, img_id)
            k = k + 1

            if args.verbose:
                draw_boxes(args, img.cpu().numpy(), recognized_boxes, recognized_scores,
                           debug_boxes, debug_scores, scale, img_id)
                
    

def test_gtdb(trained_model='weights/AMATH512_e1GTDB.pth',visual_threshold=0.6,cuda=False,verbose = False, exp_name='SSD', model_type = 512, 
                 use_char_info=False,limit = -1, cfg = "hboxes512",batch_size = 16, num_workers = 4, kernel = [1,5], padding = [3,3],neg_mining = True,
                 stride = 0.1, window = 1200,root_folder = '../../files',stitching_algo='equal',algo_threshold=30,preprocess=False,postprocess=False,gen_math=True):
    
    start = time.time()

    args = Arguments(trained_model,visual_threshold,cuda, verbose, exp_name, model_type, use_char_info,limit, cfg,
                 batch_size, num_workers, kernel, padding,neg_mining ,stride, window,root_folder,stitching_algo,
                 algo_threshold,preprocess,postprocess,gen_math)
    print("Criando imagens a partir do PDF...")
    file_name = create_images_from_pdfs(args.root_folder,args.exp_name)
    

    """
    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if os.path.exists(os.path.join(args.save_folder)):
        shutil.rmtree(os.path.join(args.save_folder))
    
    try:
        filepath=os.path.join(args.log_dir, args.exp_name + "_" + str(round(time.time())) + ".log")
        print('Logging to ' + filepath)
        logging.basicConfig(filename=filepath,
                            filemode='w', format='%(process)d - %(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)


        gpu_id = 0
        if args.cuda:
            gpu_id = helpers.get_freer_gpu()
            torch.cuda.set_device(gpu_id)
            map_location = {'cuda:1':'cuda:0'}
        else:
            map_location=torch.device('cpu')

        # load net
        num_classes = 2 # +1 background

        # initialize SSD
        net = build_ssd(args, 'test', exp_cfg[args.cfg], gpu_id, args.model_type, num_classes)

        logging.debug(net)
        if args.cuda:
            net.to(gpu_id)
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(args.trained_model, map_location=map_location))
        net.eval()
        logging.debug('Finished loading model!')

        dataset = GTDBDetection(args, args.test_data, split='test',
                                transform=BaseTransform(args.model_type, (246,246,246)),
                                target_transform=GTDBAnnotationTransform())

        if args.cuda:
            net = net.to(gpu_id)
            cudnn.benchmark = True

        # evaluation
        test_net_batch(args, net, gpu_id, dataset,
                    BaseTransform(args.model_type, (246,246,246)),
                    thresh=args.visual_threshold)
        #stitch
        print('Iniciando Stitch...')
        stitch(args)
        
    
        # Visualize
        print('Fazendo Anotações')
        math_file, img_subdir =  visualize(img_dir =args.img_dir, out_dir=args.output_dir_annot,math_dir=args.math_dir_annot)
        """
    
    
    
    try:

        print('Fazendo Anotações')
        math_file, img_subdir =  visualize(img_dir =args.img_dir, out_dir=args.output_dir_annot,math_dir=args.math_dir_annot)

        #Crop e Resize
        print('Cortando as Imagens e Fazendo Resize...')
        crop_resize(args.dataset_root,args.save_crop,args.output_dir,file_name)

        #Gerando Látex
        #print('Gerando arquivo TXT com Látex...')
        #image2latex(args.save_crop,args.dataset_root)

    
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
    
    end = time.time()
    logging.debug('Total time taken ' + str(datetime.timedelta(seconds=end-start)))
    logging.debug("Testing done!")

    return args.home_anno

def image2latex(image_dir,output_dir):
    f = open(os.path.join(output_dir,'latex.txt'),'w')
    model = pix2tex.LatexOCR()
    for name in os.listdir(image_dir):
        file = os.path.join(image_dir,name)
        img = Image.open(file)
        math = model(img)
        f.write(math+'\n')
    
    f.close()


