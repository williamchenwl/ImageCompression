import os

print('Please Input The Version Name:')

version = input()

ins1 = 'python train.py --batch-size 64 --train /data/g_data --max-epoch 50 --version {}'.format(version)

ins2 = 'python train.py --train /data/unlabeled2017 --batch-size 32 --max-epoch 100 --version {} --checkpoint 00000050'.format(version)

ins3 = 'python train.py --train /data/train2017 --batch-size 32--max-epoch 150 --version {} --checkpoint 00000100'.format(version)

ins4 = 'python train.py --train /data/train --max-epoch 200 --batch-size 32 --version {} --checkpoint 0000150'.format(version)

#test_path = '~/testbench/{}'.format(version)
#os.system('mkdir -p {}/model'.format(test_path))
#os.system('cp -r test/* {}'.format(test_path))
#os.system('cp network.py {}'.format(test_path))


os.system(ins1)
#os.system('cp -r checkpoint/epoch_00000050 {}/model/50'.format(test_path))
#os.system('python {}/test.py -m 50 -d Kodak')

os.system(ins2)
#os.system('cp -r checkpoint/epoch_00000100 {}/model/100'.format(test_path))
#os.system('python {}/test.py -m 100 -d Kodak')

os.system(ins3)
#os.system('cp -r checkpoint/epoch_00000150 {}/model/150'.format(test_path))
#os.system('python {}/test.py -m 150 -d Kodak')

os.system(ins4)
#os.system('cp -r checkpoint/epoch_00000200 {}/model/200'.format(test_path))
#os.system('python {}/test.py -m 200 -d Kodak')
