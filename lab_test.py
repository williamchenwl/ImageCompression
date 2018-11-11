import os
import argparse
import metric

def mean(a):
    return sum(a) / len(a)

def test_validation(model_path, version, root):

    os.system('mkdir -p codes_val/{}'.format(version))
    os.system('mkdir -p res_val/{}'.format(version))
    bpp = [] 
    psnr = []
    ssim = []
    for filename in os.listdir(root):
        original = os.path.join(root, filename)
        filename = filename[:-4]
        os.system('mkdir -p res_val/{}/{}'.format(version, filename))
        os.system('python encoder.py --model {}/encoder.pth --input {} --output codes_val/{}/{} '.format(model_path, original, version, filename))
        os.system('python decoder.py --model {}/decoder.pth --input codes_val/{}/{}.npz --output res_val/{}/{} '.format(model_path, version, filename, version, filename))
        codes = 'codes_val/{}/{}.npz'.format(version, filename)
        compared = 'res_val/{}/{}/00.png'.format(version, filename)
        bpp.append(utils.calc_bpp(codes, original))
        psnr.append(metric.psnr(original, compared))
        ssim.append(metric.msssim(compared, original))
    return mean(bpp), mean(psnr), mean(ssim)


def test_kodak(version, model_path):

    os.system('mkdir -p codes/{}'.format(version))
    os.system('mkdir -p res/{}'.format(version))

    for i in range(24):
        j = i + 1
        if j >= 10:
            n_id = str(j)
        else:
            n_id = '0' + str(j)

        filename = '/home/williamchen/Dataset/Kodak/kodim' + n_id + '.png'
        os.system('mkdir -p res/{}/{}'.format(version, n_id))
        os.system('python encoder.py --model {}/encoder.pth --input {} --output codes/{}/{}'.format(model_path, filename, version, n_id))
        print("encoded {}.npz".format(n_id))
        os.system('python decoder.py --model {}/decoder.pth --input codes/{}/{}.npz --output res/{}/{}'.format(model_path, version, n_id, version, n_id))

def test_jpg(level):
    for i in range(24):
        j = i + 1
        if j >= 10:
            n_id = str(j)
        else:
            n_id = '0' + str(j)
        res_path = 'jpg_res/{:d}/{}'.format(level, n_id)
        os.system('mkdir -p jpg_res/{:d}/{}'.format(level, n_id))
        filename = '/home/williamchen/Dataset/Kodak/kodim' + n_id + '.png'
        os.system('convert {} -quality {:d} -sampling-factor 4:2:0 {}/00.jpg'.format(filename, level, res_path))

def get_psnr(res_path, jpeg=False):
    psnr = []
    for i in range(24):
        j = i + 1
        if j >= 10:
            n_id = str(j)
        else:
            n_id = '0' + str(j)
        original = '/home/williamchen/Dataset/Kodak/kodim' + n_id + '.png'
        if not jpeg:
            compared = '{}/{}/00.png'.format(res_path, n_id)
        else:
            compared = '{}/{}/00.jpg'.format(res_path, n_id)
        psnr.append(metric.psnr(original, compared))
    return psnr

def get_ssim(res_path, jpeg=False):
    ssim = []
    for i in range(24):
        j = i + 1
        if j >= 10:
            n_id = str(j)
        else:
            n_id = '0' + str(j)
        original = '/home/williamchen/Dataset/Kodak/kodim' + n_id + '.png'
        compared = '{}/{}/00.png'.format(res_path, n_id)
        if jpeg:
            compared = '{}/{}/00.jpg'.format(res_path, n_id)
        ssim.append(metric.msssim(compared, original))
    return ssim

import utils

def get_bpp(res_path, jpeg=False):
    bpp= []
    for i in range(24):
        j = i + 1
        if j >= 10:
            n_id = str(j)
        else:
            n_id = '0' + str(j)
        original = '/home/williamchen/Dataset/Kodak/kodim' + n_id + '.png'
        if jpeg:
            compared = '{}/{}/00.jpg'.format(res_path, n_id)
        else:
            compared = '{}/{}.npz'.format(res_path, n_id)
        bpp.append(utils.calc_bpp(compared, original))
    return bpp

if __name__ == '__main__':

    #test_kodak('bpp-0.02-1', 'checkpoint/bpp-0.02-1/epoch_00000001')
    test_jpg(2)
    print(get_bpp('jpg_res/{:d}'.format(2), jpeg=True))
