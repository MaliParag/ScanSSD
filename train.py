# Sample command
# python3 train.py --dataset GTDB --dataset_root /home/psm2208/data/GTDB/ --cuda True --visdom True
# --batch_size 32 --num_workers 4 --layers_to_freeze 0 --save_folder weights_512 --model_type 512

from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from utils import helpers

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

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
parser.add_argument('--cuda', default=True, type=bool,
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
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--layers_to_freeze', default=20, type=float,
                    help='Number of VGG16 layers to freeze')
parser.add_argument('--model_type', default=300, type=int,
                    help='Type of ssd model, ssd300 or ssd512')
parser.add_argument('--suffix', default="_10", type=str,
                    help='Stride % used while generating images or dpi from which images was generated or some other identifier')
parser.add_argument('--type', default="processed_train", type=str,
                    help='Type of image set to use. This is list of file names, one per line')
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
parser.add_argument('--neg_mining', default=True, type=bool,
                    help='Whether or not to use hard negative mining with ratio 1:3')

args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():

    cfg = exp_cfg[args.cfg]
    dataset = GTDBDetection(args,
                            transform=SSDAugmentation(cfg['min_dim'],
                                                      MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    gpu_id = 0

    if args.cuda:
        gpu_id = helpers.get_freer_gpu()
        print('Using GPU with id ', gpu_id)
        torch.cuda.set_device(gpu_id)

    ssd_net = build_ssd(args, 'train', cfg, gpu_id, cfg['min_dim'], cfg['num_classes'])

    net = ssd_net
    print(net)

    ct = 0
    # freeze first few layers
    for child in net.vgg.children():
        if ct >= args.layers_to_freeze:
            break

        child.requires_grad = False
        ct += 1

    if args.cuda:
        #net = torch.nn.DataParallel(ssd_net)
        net = net.to(gpu_id)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load("base_weights/" + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    # if args.cuda:
    #     net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(args, cfg, 0.5, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    min_total_loss = float('inf')
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD ' + str(args.model_type) + ' on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', viz, vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', viz, vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):

        t0 = time.time()


        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            epoch += 1
            update_vis_plot(epoch, loc_loss, viz, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
             batch_iterator = iter(data_loader)
             images, targets = next(batch_iterator)
        #images, targets = next(batch_iterator)

        if args.cuda:
            #images = Variable(images.cuda())
            #targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = args.alpha * loss_l + loss_c
        loss.backward()
        optimizer.step()
        loc_loss += loss_l.item()#data[0]
        conf_loss += loss_c.item()#data[0]

        t1 = time.time()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

        if args.visdom:
            update_vis_plot(iteration, loss_l.item(), viz, loss_c.item(),
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 50 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(),
                       os.path.join(
                        args.save_folder, 'ssd' + str(args.model_type) + args.dataset +
                        repr(iteration) + '.pth'))

        elif loss.item() < min_total_loss:
            min_total_loss = loss.item()
            torch.save(ssd_net.state_dict(),
                       os.path.join(
                           args.save_folder, 'best_ssd' + str(args.model_type) + args.dataset +
                           repr(iteration) + '.pth'))

    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, viz, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
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

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    train()
