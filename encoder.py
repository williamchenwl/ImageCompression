import argparse

import numpy as np
from scipy.misc import imread, imresize, imsave

import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', '-m', required=True, type=str, help='path to model')
parser.add_argument(
    '--input', '-i', required=True, type=str, help='input image')
parser.add_argument(
    '--output', '-o', required=True, type=str, help='output codes')
parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument(
    '--iterations', type=int, default=16, help='unroll iterations')
args = parser.parse_args()

image = imread(args.input, mode='RGB')
image = torch.from_numpy(
    np.expand_dims(
        np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)), 0))
batch_size, input_channels, height, width = image.size()

assert height % 32 == 0 and width % 32 == 0

image = Variable(image, volatile=True)

import network

encoder = network.EncoderCell()
binarizer = network.Binarizer()
decoder = network.DecoderCell()

encoder.eval()
binarizer.eval()
decoder.eval()

encoder.load_state_dict(torch.load(args.model))
binarizer.load_state_dict(
    torch.load(args.model.replace('encoder', 'binarizer')))
decoder.load_state_dict(torch.load(args.model.replace('encoder', 'decoder')))

'''
if args.cuda:
    encoder = encoder.cuda()
    binarizer = binarizer.cuda()
    decoder = decoder.cuda()

    image = image.cuda()
'''

codes = []

res = image - 0.5

iterations = 1

for iters in range(iterations):
    
    encoded1, encoded4 = encoder(res) 
    
 
    code1,  code4 = binarizer(encoded1, encoded4)

    output = decoder(code1, code4)

    res = res - output

    codes.append(code1.data.cpu().numpy())
    codes.append(code4.data.cpu().numpy())
    

    print('Iter: {:02d}; Loss: {:.06f}'.format(iters, res.data.abs().mean()))


#torch.save(code.data, 'test.pth')

for i in range(len(codes)):
    codes[i] = (np.stack(codes[i]).astype(np.int8) + 1) // 2

#np.save("new.npz", codes.reshapie(-1))

#print(codes.size())

#torch.save(torch.from_numpy(codes),'1.pth')
export = []
for i in range(len(codes)):
    export.append(np.packbits(codes[i].reshape(-1)))

from utils import CABAC_encoder

size = 0

for i in range(len(export)):
    size += CABAC_encoder(export[i])

filex = open('CABAC.txt','a+')
filex.write(str(size) + '\n')

np.savez_compressed(args.output, 
                    shape1 = codes[0].shape, codes1=export[0],
                    shape2 = codes[1].shape, codes2=export[1],
                    )
