#!usr/bin/python3

import numpy as np
from scipy.misc import imread, imresize, imsave
import torch
from torch.autograd import Variable
import network
import os
from PIL import Image
import utils
from lab_test import mean
import lab_test
import metric

decoder = network.DecoderCell()
decoder.eval()

def load_model(model):
    model = model + '/encoder.pth'
    decoder.load_state_dict(
        torch.load(model.replace('encoder', 'decoder'))        
    )


def encode_an_image(image, filename):
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image)
    batch_size, input_channels, height, width = image.size()
    image = Variable(image, volatile=True)
    res = image - 0.5
    encoded1, encoded4 = encoder(res)
    code1, code4 = binarizer(encoded1, encoded4)
    codes = []
    codes.append(code1.data.cpu().numpy())
    codes.append(code4.data.cpu().numpy())
    for i in range(len(codes)):
        codes[i] = (np.stack(codes[i]).astype(np.int8) + 1) // 2
    export = []
    for i in range(len(codes)):
        export.append(np.packbits(codes[i].reshape(-1)))
    np.savez_compressed(   filename,
                            shape1 = codes[0].shape,
                            codes1 = export[0],
                            shape2 = codes[1].shape,
                            codes2 = export[1],
                        )

def decode_an_image(filename):
    
    content = np.load(filename)
    codes = []
    for i in range(2):
        codes.append(np.unpackbits(content['codes{}'.format(str(i+1))]))
        codes[i] = np.reshape(codes[i], content['shape{}'.format(str(i+1))])
        codes[i] = codes[i].astype(np.float32)
        codes[i] = codes[i] * 2 - 1
        codes[i] = torch.from_numpy(codes[i])
    
    batch_size, channels, height, width = codes[0].size()
    height = height * 4
    width = width * 4

    for i in range(len(codes)):
        codes[i] = Variable(codes[i], volatile=True)
    
    image = torch.zeros(1, 3, height, width) + 0.5

    output = decoder(codes[0], codes[1])
    image = image + output.data.cpu()
    image = image.numpy().clip(0, 1) * 255.0
    image = image.astype(np.uint8)
    image = np.squeeze(image)
    image = np.transpose(image,(1,2,0))
    return image

def decode_image_with_padding(input_path, output_path, filename):
    shape_f = open(os.path.join(input_path,filename) + '.shape')
    size = shape_f.readlines()
    print(size)
    size = tuple(size)
    width, height = size
    width, height = int(width), int(height)
    #padded = Image.open(filename+'.png'+'padded')
    padded = decode_an_image(os.path.join(input_path, filename) + '.npz')
    #print(padded.shape)
    padded = Image.fromarray(padded)
    padded.crop((0, 0, width, height))
    to_save = Image.new('RGB',(width, height))
    to_save.paste(padded)
    to_save.save(os.path.join(output_path, filename) + '.png','png')

def encode_image_with_padding(input_path, filename, output_path):
    
    image = Image.open(os.path.join(input_path,filename)).convert('RGB')
    
    width, height = image.size
    
    nh, nw = height, width

    if nh % 16 != 0:
        nh = ((height // 16) + 1) * 16

    if nw % 16 != 0:
        nw = ((height // 16) + 1) * 16

    padded = Image.new('RGB',(nw, nh))
    padded.paste(image)
    shape_path = os.path.join(output_path,filename[:-3]) + 'shape' 
    shape_f = open(shape_path,'w')
    shape_f.writelines([str(width) + '\n', str(height)])
    shape_f.close()
    
    encode_an_image(padded, os.path.join(output_path, filename[:-4]))

    #padded.save(filename+'padded','png')

def test_valid(model_path, version, root):
    os.system('mkdir -p codes_val/{}'.format(version))
    os.system('mkdir -p res_val/{}'.format(version))
    bpp = []
    psnr = []
    ssim = []
    load_model(model_path)
    for filename in os.listdir(root):
        original = os.path.join(root, filename)
        #filename = filename[:-4]
        codes_path = 'codes_val/{}'.format(version)
        output_path = 'res_val/{}/{}'.format(version, filename)
        os.system('mkdir -p {}'.format(output_path))
        encode_image_with_padding(root, filename, codes_path)
        codes = codes_path + '/' + filename[:-4] + '.npz'
        filename = filename[:-4]
        decode_image_with_padding(codes_path, output_path, filename)
        compared = output_path + '/' + filename + '.png'
        bpp.append(utils.calc_bpp(codes, original))
        psnr.append(metric.psnr(original, compared))
        ssim.append(metric.msssim(compared, original))
    return mean(bpp), mean(psnr), mean(ssim)

import argparse
from glob import glob

if __name__ == '__main__':
    
    load_model('entropy-1/saved1')
    for filename in glob('image/*.npz'):
        filename = filename[:-4]
        decode_image_with_padding('image',filename, 'image')
    
    #encode_image_with_padding('.', '1.png', 'res')
    #decode_image_with_padding('res','res', '1')
    #test_valid('entropy-1/saved1', 'test', '/home/williamchen/Dataset/Kodak')
