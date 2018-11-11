import os 

resdir = 'exp_res/'
encoded = 'codes/'
decoded = 'res1'
origin = '~/Dataset/Kodak/kodim'

def calc():
    ssim = resdir + 'ssim.txt'
    psnr = resdir + 'psnr.txt'
    bpp = resdir + 'bpp'
    os.system('mkdir -p {}'.fromat(bpp))
    os.system('mkdir -p {}'.format(ssim))
    os.system('mkdir -p {}'.format(psnr))
    os.system('echo -n "" > {}'.format(ssim))
    os.system('echo -n "" > {}'.format(psnr))
    os.system('echo -n "" > {}'.format(bpp))
    os.system('echo -n `python metric.py -m ssim -o `')

