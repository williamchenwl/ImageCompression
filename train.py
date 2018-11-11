from tensorboard_logger import configure, log_value
import utils
import time
import os
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms
from test import test_dataset

parser = argparse.ArgumentParser()

parser.add_argument(
            '--batch-size', '-N', type=int, default=64, help='batch size')

parser.add_argument(
            '--version', '-v', type=str, required=True, help='exp_place'
)
parser.add_argument(
            '--train', '-f', default='/data/g_data', type=str, help='folder of training images')
parser.add_argument(
            '--max-epochs', '-e', type=int, default=100, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
# parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument(
            '--iterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')
parser.add_argument('--mat', type=str, default='', help='load from mat file')
args = parser.parse_args()

import dataset

train_transform = transforms.Compose([
            transforms.RandomCrop((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])

def load_from_image_folder():
    train_set = dataset.ImageFolder(root=args.train, transform=train_transform)
    train_loader = data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    return train_set, train_loader

def load_from_mat_file():
    train_set = dataset.MatFile(filename=args.mat, transform=train_transform)
    train_loader = data.DataLoader(dataset=train_set, batch_size=args.bath_size, shuffle=True, num_workers=1)
    return train_set, train_loader

if args.mat == '':
    train_set, train_loader = load_from_image_folder()
else:
    train_set, train_loader = load_from_mat_file()

print('total images: {}; total batches: {}'.format(
        len(train_set), len(train_loader)))

import network

encoder = network.EncoderCell().cuda()
binarizer = network.Binarizer().cuda()
decoder = network.DecoderCell().cuda()

solver = optim.Adam(
    [
        {
            'params': encoder.parameters()
        },
        {
            'params': binarizer.parameters()
        },
        {
            'params': decoder.parameters()
        }
    ]
    , lr = args.lr
)

def resume(epoch=None):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'

    save_path = 'checkpoint/{}/epoch_{:08}'.format(args.version, epoch)

    encoder.load_state_dict(
        torch.load(os.path.join(save_path,'encoder.pth')))
    binarizer.load_state_dict(
        torch.load(os.path.join(save_path,'binarizer.pth')))
    decoder.load_state_dict(
        torch.load(os.path.join(save_path,'decoder.pth')))

def save(index, epoch=True):

    if not os.path.exists('checkpoint/{}'.format(args.version)):
        os.system('mkdir -p checkpoint/{}'.format(args.version))

    save_path = 'checkpoint/{}/epoch_{:08}'.format(args.version, index)

    if index > 2 and index % 20 != 1:
        os.system('rm -r checkpoint/{}/epoch_{:08}'.format(args.version, index - 1))
    if not os.path.exists(save_path):
        os.system('mkdir -p {}'.format(save_path))

    if epoch:
        s = 'epoch'
    else:
        s = 'iter'


    torch.save(encoder.state_dict(), os.path.join(save_path,'encoder.pth'))

    torch.save(binarizer.state_dict(), os.path.join(save_path,'binarizer.pth'))

    torch.save(decoder.state_dict(), os.path.join(save_path,'decoder.pth'))
    
    return save_path

scheduler = LS.MultiStepLR(solver, milestones=[3, 10, 20, 50, 100], gamma=0.5)

last_epoch = 0

if args.checkpoint:
    resume(args.checkpoint)
    last_epoch = args.checkpoint
    scheduler.last_epoch = last_epoch - 1

def criterion(res):
    
    loss = res.pow(2).mean()

    loss = loss + 0.2 * loss * utils.tensor_entropy(res)

    return loss

configure('/home/amax/william/runs/{}'.format(args.version), flush_secs=3)

for epoch in range(last_epoch + 1, args.max_epochs + 1):

    scheduler.step()

    epoch_loss = []

    rec_en1 = []
    rec_en2 = []

    for batch, data in enumerate(train_loader):

        batch_t0 = time.time()

        patches = Variable(data.cuda())

        solver.zero_grad()

        losses = []

        entropy = 0

        res = patches - 0.5

        bp_t0 = time.time()

        encoded = encoder(res)

        if epoch >= 15:
            codes = binarizer(*encoded)
        else:
            codes = encoded

        output  = decoder(*codes)

        res = res - output

       	losses.append(res.pow(2).mean())

        epoch_loss.append(res.abs().mean().data.cpu().numpy())
    
        bp_t1 = time.time()

        loss = sum(losses)

        loss.backward()

        solver.step()

        batch_t1 = time.time()

        print(
                '[TRAIN] Epoch[{}]({}/{}); Loss: {:.6f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.
            format(epoch, batch + 1,
                   len(train_loader), loss.data[0], bp_t1 - bp_t0, batch_t1 -
                   batch_t0))

        index = (epoch - 1) * len(train_loader) + batch

    final_loss = sum(epoch_loss) / len(epoch_loss)

    print('EPOCH LOSS:')
    print(final_loss)

    log_value('epoch_loss', final_loss, epoch)
    
    save_path = save(epoch)

    if epoch % 5 == 0 and epoch >= 20:
        bpp, psnr, ssim = test_dataset(save_path, str(epoch) + 'Kodak', calc_bpp = True, dataset = 'Kodak')
        log_value('bpp_k', bpp, int(epoch // 5 - 3))
        log_value('psnr_k', psnr, int(epoch // 5 - 3))
        log_value('ssim_k', ssim, int(epoch // 5 - 3))
        bpp, psnr, ssim = test_dataset(save_path, str(epoch) + 'pval', calc_bpp = True, dataset = 'pval')
        log_value('bpp', bpp, int(epoch // 5 - 3))
        log_value('psnr', psnr, int(epoch // 5 - 3))
        log_value('ssim', ssim, int(epoch // 5 - 3))

