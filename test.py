#!/usr/bin/env python3
import time
import numpy as np
from scipy.misc import imread, imresize, imsave
import torch
from torch.autograd import Variable
import os
from PIL import Image
import utils
from utils import PILImageToNumpy, padding_image
from lab_test import mean
import lab_test
import metric

import network

default_name = ['encoder', 'decoder', 'bianrizer']

class CodeC():

    def __init__(self, original_path, model_path, codes_path, save_path, decode_only = False, gpu = True):

        self.gpu = gpu 

        global encoder 
        
        encoder = network.EncoderCell()

        global binarizer 
        
        binarizer = network.Binarizer()

        global decoder 
        
        decoder = network.DecoderCell()

        if gpu:

            self.encoder = encoder.cuda()
            bianrizer = binarizer.cuda()
            decoder = decoder.cuda()

        encoder = encoder.eval()

        bianrizer = binarizer.eval()

        decoder = decoder.eval()

        self.original = original_path  

        self.codes_path = codes_path

        os.system('mkdir -p {}'.format(self.codes_path))

        self.save_path = save_path

        os.system('mkdir -p {}'.format(self.save_path))

        model = model_path + '/encoder.pth'

        if gpu:

            if decode_only == False:
                encoder.load_state_dict(torch.load(model))
                binarizer.load_state_dict(
                    torch.load(model.replace('encoder', 'binarizer')))
            decoder.load_state_dict(
                    torch.load(model.replace('encoder', 'decoder')))
            
        else:
            
            if decode_only == False:
                encoder.load_state_dict(
                    torch.load(model, 
                    map_location = lambda storage, loc:storage)
                )
                binarizer.load_state_dict(
                    torch.load(model.replace('encoder', 'binarizer'), 
                    map_location = lambda storage, loc:storage)
                )
            decoder.load_state_dict(
                    torch.load(model.replace('encoder','decoder'), 
                    map_location = lambda storage, loc:storage)
            )

    def full_pipeline(self, image):
    
        image = PILImageToNumpy(image)

        image = torch.from_numpy(image)

        batch_size, input_channels, height, width = image.size()

        image = Variable(image, volatile = True)

        res = image - 0.5
        
        if self.gpu:
            res = res.cuda()

        encoded = encoder(res)
        
        codes = binarizer(*encoded)

        output = decoder(*codes)

        image = output.data.cpu() + 0.5

        image = image.numpy().clip(0, 1) * 255.0
        
        image = image.astype(np.uint8)

        image = np.squeeze(image)

        image = np.transpose(image, (1, 2, 0))

        return image

    def full_pipeline_with_padding(self, filename):

        time_t0 = time.time()
        
        original = os.path.join(self.original, filename)

        image = Image.open(original).convert('RGB')

        image, width, height = padding_image(image, 128)

        image = self.full_pipeline(image)
        
        image = Image.fromarray(image)

        image.crop((0, 0, width, height))
        
        to_save = os.path.join(self.save_path, filename)

        to_save_image = Image.new('RGB', (width, height))

        to_save_image.paste(image)

        to_save_image.save(to_save, 'png')
        
        time_t1 = time.time()

        print('compress time is {:.4f} sec'.format(time_t1 - time_t0))

    def encode_single_image(self, image, filename):

        '''
            encode a single image

        '''

        image = PILImageToNumpy(image)

        image = torch.from_numpy(image)

        image = Variable(image, volatile=True)

        res = image - 0.5
        
        if self.gpu:
            res = res.cuda()

        encoded = encoder(res)

        codes = binarizer(*encoded)

        export = []

        if codes[-1] is None:

            codes = codes[:-1]

        codes = list(codes)

        for i in range(len(codes)):
            codes[i] = codes[i].data.cpu().numpy()
            codes[i] = (np.stack(codes[i]).astype(np.int8) + 1) // 2
            shape = codes[i].shape
            export.append(np.packbits(codes[i].reshape(-1)))
            np.savez_compressed(
                    os.path.join(self.codes_path, filename[:-4] + str(i)),
                    shape = shape,
                    codes = export[i]
            )

    def decode_single_image(self, filename):
            
        content = []

        codes = []

        for i in range(10):
            
            code_name = os.path.join(self.codes_path, filename[:-4] + str(i) + '.npz')
            if os.path.exists(code_name):
                content.append(np.load(code_name))        
            else:
                break
            
        for i in range(len(content)):
            
            codes.append(np.unpackbits(content[i]['codes']))
            codes[i] = np.reshape(codes[i], content[i]['shape'])
            codes[i] = codes[i].astype(np.float32)
            codes[i] = codes[i] * 2 - 1
            codes[i] = torch.from_numpy(codes[i])
            codes[i] = Variable(codes[i], volatile = True)
            codes[i] = codes[i].cuda()

        
        if len(codes) == 1: 

            codes.append(codes[0])
        
        image = decoder(*codes)
        image = image.data.cpu().numpy() + 0.5
        image = image * 255.0
        image = image.astype(np.uint8)
        image = np.squeeze(image)
        image = np.transpose(image, (1, 2, 0))

        return image
            
    def test_single_image(self, filename):

        time_t0 = time.time()

        original = os.path.join(self.original, filename)

        image = Image.open(original).convert('RGB')
        
        image, width, height = padding_image(image, 128)
   
        self.encode_single_image(image, filename)

        image = self.decode_single_image(filename)

        image = Image.fromarray(image)

        image.crop((0, 0, width, height))

        to_save = os.path.join(self.save_path, filename)
        
        to_save_image = Image.new('RGB', (width, height))

        to_save_image.paste(image)

        to_save_image.save(to_save, 'png')

        time_t1 = time.time()

        print('compress time is {:.4f} sec'.format(time_t1 - time_t0))


    
def test_dataset(model_path, version, calc_bpp = False, dataset = 'Kodak'):

    original_path = os.path.join('/data', dataset)

    codes_path = os.path.join(os.path.join('res', dataset), 'codes')

    res_path = os.path.join(os.path.join('res', dataset), 'pic')

    MCodeC = CodeC(original_path, model_path, codes_path, res_path)

    for filename in os.listdir(original_path):
        
        if calc_bpp == True:
            MCodeC.test_single_image(filename)
        else:
            MCodeC.full_pipeline_with_padding(filename)
    
    return compare_folder(original_path, codes_path, res_path)

def compare_folder(origin, codes, res):

    psnr = []
    ssim = []

    total_pixels = 0
    
    for filename in os.listdir(origin):

        original_i = os.path.join(origin, filename)

        res_i = os.path.join(res, filename)

        psnr.append(metric.psnr(original_i, res_i))

        ssim.append(metric.msssim(original_i, res_i))
        
        total_pixels += utils.get_pixels(original_i)
        print(psnr[-1], ssim[-1])

    total_size = utils.get_size_folder(codes)
        
    bpp = total_size / total_pixels 

    print(bpp, mean(psnr), mean(ssim))

    return bpp, mean(psnr), mean(ssim)

import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--dataset', '-d', type=str, default='Kodak')
    args = parser.parse_args()
    test_dataset(args.model, 'test', calc_bpp = True, dataset = args.dataset)
    #record = open('report.txt','a+')
    #record.write('epoch: {}, bpp : {:.4f},  psnr : {:.4f} ssim : {:.4f}'.format(args.model, bpp, psnr, ssim))

