# Sample command
# python3 train.py --dataset GTDB --dataset_root /home/psm2208/data/GTDB/
# --cuda True --visdom True --batch_size 16 --num_workers 8 --layers_to_freeze 0
# --exp_name weights_1 --model_type 512 --suffix _512 --type processed_train_512
# --cfg math_gtdb_512 --loss_fun fl --kernel 1 5 --padding 0 2 --neg_mining False

from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import argparse
from utils import helpers
import logging
import time
import datetime
from torchviz import make_dot

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def train(args):

    cfg = exp_cfg[args.cfg]
    dataset = GTDBDetection(args, args.training_data, split='train',
                            transform=SSDAugmentation(cfg['min_dim'], mean=MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    gpu_id = 0

    if args.cuda:
        gpu_id = helpers.get_freer_gpu()
        logging.debug('Using GPU with id ' + str(gpu_id))
        torch.cuda.set_device(gpu_id)

    ssd_net = build_ssd(args, 'train', cfg, gpu_id, cfg['min_dim'], cfg['num_classes'])

    logging.debug(ssd_net)

    ct = 0
    # freeze first few layers
    for child in ssd_net.vgg.children():
        if ct >= args.layers_to_freeze:
            break

        child.requires_grad = False
        ct += 1


    if args.resume:
        logging.debug('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_state_dict(torch.load(args.resume))
    else:
        vgg_weights = torch.load("base_weights/" + args.basenet)
        logging.debug('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    #visualize(ssd_net, gpu_id)

    # if args.cuda:
    #     net = net.cuda()
    step_index = 0

    if not args.resume:
        logging.debug('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

        for val in cfg['lr_steps']:
            if args.start_iter > val:
                step_index = step_index + 1

        # Saving random initialized weights
        torch.save(ssd_net.state_dict(),
                   os.path.join(
                       'weights_' + args.exp_name, 'initial_' + str(args.model_type) + args.dataset + '.pth'))

    optimizer = optim.SGD(ssd_net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    adjust_learning_rate(args, optimizer, args.gamma, step_index)

    #args, cfg, overlap_thresh, bkg_label, neg_pos
    #criterion = MultiBoxLoss(args, cfg, 0.5, 0, 3)
    criterion = MultiBoxLoss(args, cfg, args.pos_thresh, 0, 3)

    if args.cuda:
        ssd_net = torch.nn.DataParallel(ssd_net)
        # ssd_net = ssd_net.to(gpu_id)
        cudnn.benchmark = True

    ssd_net.train()

    # loss counters
    loc_loss = 0
    conf_loss = 0
    min_total_loss = float('inf')
    epoch = 0
    logging.debug('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size

    logging.debug('Training SSD on:' + dataset.name)
    logging.debug('Using the specified args:')
    logging.debug(args)


    if args.visdom:
        vis_title = args.exp_name
        vis_legend = ['Location Loss', 'Confidence Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', viz, 'Training ' + vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', viz, 'Training ' + vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    logging.debug('Training set size is ' + str(len(dataset)))

    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):

        # resume training
        ssd_net.train()

        t0 = time.time()

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(args, optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets, _ = next(batch_iterator)
        except StopIteration:
             batch_iterator = iter(data_loader)
             images, targets, _ = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        # forward
        out = ssd_net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = args.alpha * loss_l + loss_c #TODO. For now alpha should be 1. While plotting alpha is assumed to be 1
        loss.backward()
        optimizer.step()

        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        t1 = time.time()

        # Log progress
        if iteration % 10 == 0:
            logging.debug('timer: %.4f sec.' % (t1 - t0))
            logging.debug('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()))

        if args.visdom:
            update_vis_plot(iteration, loss_l.item(), viz, loss_c.item(),
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 1000 == 0:
            logging.debug('Saving state, iter:' + str(iteration))
            torch.save(ssd_net.state_dict(),
                       os.path.join(
                        'weights_' + args.exp_name, 'ssd' + str(args.model_type) + args.dataset +
                        repr(iteration) + '.pth'))


        if iteration!=0 and (iteration % epoch_size == 0):
            epoch += 1

            torch.save(ssd_net.state_dict(),
                       os.path.join(
                           'weights_' + args.exp_name, 'epoch_ssd' + str(args.model_type) + args.dataset +
                           repr(epoch) + '.pth'))


            train_loss = loc_loss + conf_loss
            update_vis_plot(epoch, loc_loss, viz, conf_loss, epoch_plot, None,
                            'append', epoch_size)

            if args.validation_data != '':
                # Validate data
                validation_loss = validate(args, ssd_net, criterion, cfg)

                if epoch == 1:
                    validation_plot = create_validation_plot(epoch, validation_loss,
                                                             'Epoch', 'Loss', viz, 'Validating ' + vis_title,
                                                             ['Validation'])
                else:
                    update_validation_plot(epoch, validation_loss, viz,
                                           validation_plot, 'append')

                if validation_loss < min_total_loss:
                    min_total_loss = validation_loss
                    torch.save(ssd_net.state_dict(),
                               os.path.join(
                                   'weights_' + args.exp_name, 'best_ssd' + str(args.model_type) + args.dataset +
                                   repr(iteration) + '.pth'))

            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0

    torch.save(ssd_net.state_dict(),
               args.exp_name + '' + args.dataset + '.pth')

    logging.debug("Final weights are saved at " + args.exp_name + '' + args.dataset + '.pth')

def validate(args, net, criterion, cfg):

    validation_batch_size = 1
    try:
        # Turn off learning. Go to testing phase
        net.eval()

        dataset = GTDBDetection(args, args.validation_data, split='validate',
                                transform=SSDAugmentation(cfg['min_dim'], mean=MEANS))

        data_loader = data.DataLoader(dataset, validation_batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=False, collate_fn=detection_collate,
                                      pin_memory=True)

        total = len(dataset)
        done = 0
        loc_loss = 0
        conf_loss = 0

        start = time.time()

        for batch_idx, (images, targets, ids) in enumerate(data_loader):

            done = done + len(images)
            logging.debug('processing {}/{}'.format(done, total))

            if args.cuda:
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            y = net(images)  # forward pass

            loss_l, loss_c = criterion(y, targets)
            loc_loss += loss_l.item()  # data[0]
            conf_loss += loss_c.item()  # data[0]

        end = time.time()
        logging.debug('Time taken for validation ' + str(datetime.timedelta(seconds=end - start)))

        return (loc_loss + conf_loss) / (total/validation_batch_size)
    except Exception as e:
        logging.error("Could not validate", exc_info=True)
        return 0

def adjust_learning_rate(args, optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def visualize(ssd_net, gpu_id):
    x = np.zeros((300,300,3))
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = x.to(gpu_id,  dtype=torch.float)
    ssd_net.eval()
    y = ssd_net(x)
    make_dot(y[0], params=dict(ssd_net.named_parameters())).render(filename='ssd_net')


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def create_validation_plot(epoch, validation_loss, _xlabel, _ylabel, viz, _title, _legend):
    return viz.line(
        X=torch.ones((1, 1)).cpu() * epoch,
        Y=torch.Tensor([validation_loss]).unsqueeze(0).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def create_vis_plot(_xlabel, _ylabel, viz, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, len(_legend))).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_validation_plot(epoch, validation_loss,
                           viz, window, update_type):

    viz.line(
        X=torch.ones((1, 1)).cpu() * epoch,
        Y=torch.Tensor([validation_loss]).unsqueeze(0).cpu(),
        win=window,
        update=update_type
    )


def update_vis_plot(iteration, loc, viz, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )

def init_args():
    '''
    Read arguments and initialize directories
    :return: args
    '''

    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')
    train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset', default='GTDB', choices=['GTDB'],
                        type=str, help='choose GTDB')
    parser.add_argument('--dataset_root', default=GTDB_ROOT,
                        help='Dataset root directory path')
    parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                        help='Pretrained base model')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='Resume training at this iter')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in data loading')
    parser.add_argument('--cuda', default=False, type=bool,
                        help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='Alpha for the multibox loss')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--visdom', default=False, type=bool,
                        help='Use visdom for loss visualization')
    parser.add_argument('--exp_name', default='math_detector',  # changed to exp_name from --save_folder
                        help='It is the name of the experiment. Weights are saved in the directory with same name.')
    parser.add_argument('--layers_to_freeze', default=20, type=float,
                        help='Number of VGG16 layers to freeze')
    parser.add_argument('--model_type', default=300, type=int,
                        help='Type of ssd model, ssd300 or ssd512')
    parser.add_argument('--suffix', default="_10", type=str,
                        help='Stride % used while generating images or dpi from which images was generated or some other identifier')
    parser.add_argument('--training_data', default="training_data", type=str,
                        help='Training data to use. This is list of file names, one per line')
    parser.add_argument('--validation_data', default="", type=str,
                        help='Validation data to use. This is list of file names, one per line')
    parser.add_argument('--use_char_info', default=False, type=bool,
                        help='Whether to use char position info and labels')
    parser.add_argument('--cfg', default="ssd512", type=str,
                        help='Type of network: either gtdb or math_gtdb_512')
    parser.add_argument('--loss_fun', default="fl", type=str,
                        help='Type of loss: either fl (focal loss) or ce (cross entropy)')
    parser.add_argument('--kernel', default="3 3", type=int, nargs='+',
                        help='Kernel size for feature layers: 3 3 or 1 5')
    parser.add_argument('--padding', default="1 1", type=int, nargs='+',
                        help='Padding for feature layers: 1 1 or 0 2')
    parser.add_argument('--neg_mining', default=False, type=bool,
                        help='Whether or not to use hard negative mining with ratio 1:3')
    parser.add_argument('--log_dir', default="logs", type=str,
                        help='dir to save the logs')
    parser.add_argument('--stride', default=0.1, type=float,
                        help='Stride to use for sliding window')
    parser.add_argument('--window', default=1200, type=int,
                        help='Sliding window size')
    parser.add_argument('--pos_thresh', default=0.5, type=float,
                        help='All default boxes with iou>pos_thresh are considered as positive examples')

    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            logging.warning("WARNING: It looks like you have a CUDA device, but aren't " +
                            "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists("weights_" + args.exp_name):
        os.mkdir("weights_" + args.exp_name)

    return args

if __name__ == '__main__':

    args = init_args()
    start = time.time()

    try:
        filepath=os.path.join(args.log_dir, args.exp_name + "_" + str(round(time.time())) + ".log")
        print('Logging to ' + filepath)
        logging.basicConfig(filename=filepath,
                            filemode='w', format='%(process)d - %(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

        train(args)
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

    end = time.time()
    logging.debug('Total time taken ' + str(datetime.timedelta(seconds=end - start)))
    logging.debug("Training done!")
