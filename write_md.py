import os
import lab_test

def mean(list_a):
    return sum(list_a) / len(list_a)

def create_md_file(path, bpp_mine, psnr_mine, ssim_mine, bpp_jpg, psnr_jpg, ssim_jpg):

    os.system('mkdir -p {}'.format(path))
    file_p = os.path.join(path,'res.md')
    mdfile = open(file_p, 'w')
    res = []
    res.append('MyModel: mean bpp is {:.4f}, mean psnr is {:.4f}, mean ssim is {:.4f}\n'.format(mean(bpp_mine), mean(psnr_mine), mean(ssim_mine)))
    res.append('JPEG: mean bpp is {:.4f}, mean psnr is {:.4f}, mean ssim is {:.4f}\n'.format(mean(bpp_jpg), mean(psnr_jpg), mean(ssim_jpg)))
    
    res.append('|BPP_Mine |PSNR_Mine |SSIM_Mine |BPP_JPG |PSNR_JPG |SSIM_JPG |\n')
    res.append('|----|----|----|----|-----|----|\n')
    comb = zip(bpp_mine, psnr_mine, ssim_mine,bpp_jpg, psnr_jpg, ssim_jpg)
    for i in range(len(psnr_mine)):
        str = '|{:.4f} | {:.4f} | {:.4f} | {:.4f}| {:.4f} | {:.4f} | \n'.format(
        bpp_mine[i], psnr_mine[i], ssim_mine[i], bpp_jpg[i], psnr_jpg[i], ssim_jpg[i]                
)
        res.append(str)
    mdfile.writelines(res)

def process(model, version, args, run = True):
    if run:
        lab_test.test_kodak(version, model)
        lab_test.test_jpg(int(args.jpg))
    png_path = 'res/{}'.format(version)
    jpg_path = 'jpg_res/{}'.format(args.jpg)
    bpp_mine = lab_test.get_bpp('codes/{}'.format(version))
    psnr_mine = lab_test.get_psnr(png_path)
    ssim_mine = lab_test.get_ssim(png_path)
    bpp_jpg = lab_test.get_bpp(jpg_path,jpeg=True)
    psnr_jpg = lab_test.get_psnr(jpg_path,jpeg=True)
    ssim_jpg = lab_test.get_ssim(jpg_path,jpeg=True)
    save_path = 'report/{}'.format(version)
    os.system('mkdir -p {}'.format(save_path))
    create_md_file(save_path, bpp_mine, psnr_mine, ssim_mine, bpp_jpg, psnr_jpg, ssim_jpg)

def CABAC_res():
    os.system('touch CABAC.md')
    res1 = open('CABAC.txt','r')
    size1 = res1.readlines()
    res = []
    res.append('|CABAC(kb) |Huffman(kb) |\n')
    res.append('|----|----|\n')
    i = 0
    for x in size1: 
        i += 1
        if i < 10:
            n_id = '0' + str(i)
        else:
            n_id = str(i)
        res.append('|{} |{:d} |\n'.format(x.strip('\n'), os.path.getsize('codes/entropy-1/{}.npz'.format(n_id))))
    md_file = open('CABAC.md','w')
    md_file.writelines(res)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', required=True, type=str)
    parser.add_argument('--version', '-v', required=True, type=str)
    parser.add_argument('--jpg', '-j', required=True, type=str)
    args = parser.parse_args()
    process(args.model, args.version, args)
