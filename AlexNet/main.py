# Main File

# Importing System / Python libraries
import os
import argparse
import time

# Importing pytorch stuff
import torch
import torch.nn as nn
import torch.optim as optim

# Import model functions
from model import *
from setup_data import data_loader
from utils import Utils, create_checkpoint, accuracy, custom_weight_decay

best_predc = 0.0

# Initialise and setup a parser
parser = argparse.ArgumentParser(description="AlexNet training on the ImageNet dataset.")
# Path to Data
parser.add_argument('data', metavar='DIR', help='path to dataset')
# Architecture
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet')
# Number of Epochs to run
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='total epochs to run')
# Batch Size
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size / default:256')
# Learning Rate
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
# Momentum
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for optimising process')
# Weight Decay Settings
parser.add_argument('--weightdecay', '--wd', default=1e-4, type=float, metavar='W', help='Weight decay / default: 1e-4')
# CUDA (gpu) count
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='no. of available gpu workers / default:1')
# Pretrained weights
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true', help='use pre-trained weights')
# Evaluate model performance on validation set
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Evaluate model performance on validation set')



def main():
    """
    PyTorch AlexNet implementation.
    """
    global args, best_predc
    args = parser.parse_args()
    # create model
    if args.arch == 'alexnet':
        model = alexnet(pretrained=args.pretrained)
    else:
        raise NotImplementedError

    # use CUDA
    model.cuda()
    
    # define loss and optimizer
    loss = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weightdecay)

    train_dl, val_dl = data_loader(args.data, args.batch_size, args.cuda_workers)
    
    if args.evaluate:
        validate(val_dl, model, loss)
        return

    for epoch in range(args.start_epoch, args.epochs):
        custom_weight_decay(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_dl, model, loss, optimizer, epoch)

        # evaluate on validation set
        prec1, prec5 = validate(val_dl, model, loss)

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        create_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, args.arch + '.pth')


def train(train_dl, model, loss, optimizer, epoch):
    batch_time = Utils()
    data_time = Utils()
    losses = Utils()
    top1 = Utils()
    top5 = Utils()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_dl):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(nonblocking=True)
        input = input.cuda(nonblocking=True)

        # compute output
        output = model(input)
        loss = loss(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, top_k=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 200 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_dl), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_dl, model, loss):
    batch_time = Utils()
    losses = Utils()
    top1 = Utils()
    top5 = Utils()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_dl):
        target = target.cuda(nonblocking=True)
        input = input.cuda(nonblocking=True)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = loss(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, top_k=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            # top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 200 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_dl), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


if __name__ == '__main__':
    main()