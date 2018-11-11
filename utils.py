import os
from scipy.misc import imread, imresize
import numpy as np
import torch
from PIL import Image

def tensor_entropy(tensor):
    res = tensor.mean().abs()
    res = 1 - res
    return res
    

def padding_image(image, mod):
    
    width, height = image.size

    nh, nw = height, width

    if nh % mod != 0:
        nh = ((height // mod) + 1) * mod
    if nw % mod != 0:
        nw = ((width // mod) + 1) * mod

    padded = Image.new('RGB', (nw, nh))

    padded.paste(image)

    return padded, width, height


def PILImageToNumpy(image):

    image = np.array(image)
        
    image = image.astype(np.float32) / 255.0

    image = np.transpose(image, (2, 0, 1))

    image = np.expand_dims(image, 0)

    return image


def CABAC_encoder(bitstream):
    os.system('mkdir -p tmp')
    tmp_save = "tmp/saved"
    file = open(tmp_save, 'wb')
    bitstream = list(bitstream)
    bitstream = bytes(bitstream)
    file.write(bitstream)
    file.close()
    os.system('CABAC tmp/saved')
    return os.path.getsize('CABACencoded.dat')

def calc_bpp(coded, imagep):
    print(imagep)
    image = Image.open(imagep)
    height, width = image.size
    #image = torch.from_numpy(np.expand_dims(np.transpose(image.astype(np.float32) / 255.0 , (2, 0, 1)), 0))
    #_, channels, height, width = image.size()
    size = os.path.getsize(coded)
    return size * 8 / height / width

def get_size_folder(root):

    size = 0

    for filename in os.listdir(root):

        filename = os.path.join(root, filename)

        size += os.path.getsize(filename)

    return size * 8

def get_pixels(filename):

    image = Image.open(filename)
    height, width = image.size
    return height * width 


def padding_for(imagep):
    image = imread(imagep)
    image = torch.from_numpy(np.expand_dims(np.transpose(image.astype(np.float32) / 255.0 , (2, 0, 1)), 0))
    _, _, height, width = image.size()
    new_height = height // 16 * 16
    new_width = width // 16 * 16
    tmp_image = image.numpy()
    tmp_image = imresize(tmp_image, (new_height, new_width))

    return tmp_image, height, width

def paddingx(numpy_file, num):

   _, height, width = torch.from_numpy(numpy_file).size()

   new_height, new_width = height // num * num, width // num * num

   tmp_image = imresize(numpy_file, (new_height, new_width))

   return tmp_image, height, width



if __name__ == '__main__':
   idx = input()
   print(calc_bpp('codes/{}.npz'.format(idx), 'res1/{}/00.png'.format(idx)))
