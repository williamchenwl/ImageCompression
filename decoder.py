import os
import argparse

import numpy as np
from scipy.misc import imread, imresize, imsave

import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, help='path to model')
parser.add_argument('--input', required=True, type=str, help='input codes')
parser.add_argument('--output', default='.', type=str, help='output folder')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument(
    '--iterations', type=int, default=16, help='unroll iterations')
args = parser.parse_args()

content = np.load(args.input)

codes = []

for i in range(2):
    codes.append(np.unpackbits(content['codes{}'.format(str(i+1))]))
    #print(codes.size())

    codes[i] = np.reshape(codes[i], content['shape{}'.format(str(i+1))]).astype(np.float32) * 2 - 1

    codes[i] = torch.from_numpy(codes[i])

batch_size, channels, height, width = codes[0].size()

height = height * 4
width = width * 4

for i in range(len(codes)):
    codes[i] = Variable(codes[i], volatile=True)

import network

decoder = network.DecoderCell()
decoder.eval()

decoder.load_state_dict(torch.load(args.model))

if args.cuda:
    decoder = decoder.cuda()

    for i in range(len(codes)):
        codes[i] = codes[i].cuda()

image = torch.zeros(1, 3, height, width) + 0.5

iterations = 1

for iters in range(iterations):

    output = decoder(codes[0], codes[1])

    image = image + output.data.cpu()

    imsave(
        os.path.join(args.output, '{:02d}.png'.format(iters)),
        np.squeeze(image.numpy().clip(0, 1) * 255.0).astype(np.uint8)
        .transpose(1, 2, 0))
