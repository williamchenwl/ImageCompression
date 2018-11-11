import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import Sign
import numpy as np

NUM_FEAT1 = 1   #32 * 32 * 1
NUM_FEAT2 = 4   #16 * 16 * 4
NUM_FEAT3 = 16   #8 * 8 * 16
NUM_FEAT4 = 64  #4 * 4 * 64

class BottleNeck(nn.Module):

    def __init__(self, in_channels, filter_size):

        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(
                in_channels, 
                filter_size, 
                kernel_size = 1,
                stride = 1,
                bias = False
        )

        self.conv2 = nn.Conv2d(
                filter_size, 
                filter_size,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = False
        )

        self.conv3 = nn.Conv2d(
                filter_size, 
                in_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = False
        )

        self.relu = nn.LeakyReLU(0.02, inplace = True)


    def forward(self, x):

        identity = x

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.add(identity, self.conv3(x))

        return self.relu(x)

class ResBlock(nn.Module):

    def __init__(self,in_channels,filter_size):

        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            filter_size,
            kernel_size=3,
            padding=1,
            stride=1,
            bias = False
        )
        init.xavier_normal(self.conv1.weight, np.sqrt(2.0))
        self.conv2 = nn.Conv2d(
            filter_size,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias = False
        )
        init.xavier_normal(self.conv2.weight, np.sqrt(2.0))
        self.relu = nn.LeakyReLU(0.02, inplace = True)
    def forward(self, input):
        res = input
        x = self.relu(self.conv1(input))
        x = res + self.conv2(x)
        return self.relu(x)

class DownsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.downsample = nn.Conv2d(
                in_channels, 
                out_channels,
                kernel_size = 3,
                padding = 1,
                stride = 2,
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, stride = 1)
        self.relu = nn.LeakyReLU(0.02, inplace = True)
        #init.xavier_normal(self.conv1.weight, np.sqrt(2))
        #init.xavier_normal(self.conv2.weight, np.sqrt(2))
        #init.xavier_normal(self.downsample.weight, np.sqrt(2))

    def forward(self, input):
        res = input 
        #x = F.relu(self.conv1(input))
        #x = self.downsample(res) + self.conv2(x)
        return self.relu(self.downsample(res))

class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * 4, kernel_size = 3, padding = 1, stride = 1)
        self.upsample = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, padding = 0, stride = 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, stride = 1)
        #init.xavier_normal(self.conv1.weight, np.sqrt(2))
        #init.xavier_normal(self.conv2.weight, np.sqrt(2))
        #init.xavier_normal(self.upsample.weight, np.sqrt(2))
        self.relu = nn.LeakyReLU(0.02, inplace = True)

    def forward(self, input):
        x = input
        res = x
        x = self.relu(self.conv1(x))
        x = F.pixel_shuffle(x, 2)
        x = F.pixel_shuffle(self.upsample(res), 2) + self.conv2(x)
        return self.relu(x)


class EncoderCell(nn.Module):

    def __init__(self):

        super(EncoderCell, self).__init__()
        self.relu = nn.LeakyReLU(0.02, inplace = True)
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=1)
        self.branch1 = nn.Sequential(
            DownsampleBlock(64, 64),
            BottleNeck(64, 64),
            BottleNeck(64, 64),
            BottleNeck(64, 64),
            DownsampleBlock(64, 128),
            nn.Conv2d(128, NUM_FEAT1, kernel_size = 1, padding = 0, stride = 1)
        )
        
        self.branch2 = nn.Sequential(
            DownsampleBlock(64, 64),
            DownsampleBlock(64, 128),
            BottleNeck(128, 128),
            BottleNeck(128, 128),
            BottleNeck(128, 128),
            BottleNeck(128, 128),
            DownsampleBlock(128, 256),
            nn.Conv2d(256, NUM_FEAT2, kernel_size = 1, padding = 0, stride = 1)
        )

        self.branch3 = nn.Sequential(
            DownsampleBlock(64, 64),
            DownsampleBlock(64, 128),
            DownsampleBlock(128, 256),
            BottleNeck(256, 256),
            BottleNeck(256, 256),
            BottleNeck(256, 256),
            BottleNeck(256, 256),
            BottleNeck(256, 256),
            DownsampleBlock(256, 512),
            nn.Conv2d(512, NUM_FEAT3, kernel_size = 1, padding = 0, stride = 1)
        )

        self.branch4 = nn.Sequential(
            DownsampleBlock(64, 64),
            DownsampleBlock(64, 128),
            DownsampleBlock(128, 256),
            DownsampleBlock(256, 512),
            BottleNeck(512, 512),
            BottleNeck(512, 512),
            BottleNeck(512, 512),
            BottleNeck(512, 512),
            BottleNeck(512, 512),
            BottleNeck(512, 512),
            DownsampleBlock(512, 1024),
            nn.Conv2d(1024, NUM_FEAT4, kernel_size = 1, padding = 0, stride = 1)
        )

    def forward(self, input):

        x = self.relu(self.conv0(input))
        res1 = F.tanh(self.branch1(x))
        res2 = F.tanh(self.branch2(x))
        res3 = F.tanh(self.branch3(x))
        res4 = F.tanh(self.branch4(x))
        return res1, res2, res3, res4


class Binarizer(nn.Module):

    def __init__(self):
        super(Binarizer, self).__init__()
        #self.conv1 = nn.Conv2d(NUM_FEAT1, NUM_FEAT1, kernel_size=1, bias=False)
        #self.conv4 = nn.Conv2d(NUM_FEAT2, NUM_FEAT2, kernel_size=1, bias=False)
        self.sign = Sign()

    def forward(self, feat1, feat2, feat3, feat4):
        #feat1 = self.conv1(feat1)
        #feat1 = F.tanh(feat1)
        return self.sign(feat1), self.sign(feat2), self.sign(feat3), self.sign(feat4)

class DecoderCell(nn.Module):

    def __init__(self):
        super(DecoderCell, self).__init__()
        self.relu = nn.LeakyReLU(0.02, inplace = True)
        self.branch1 = nn.Sequential(
            nn.Conv2d(NUM_FEAT1, 128, kernel_size = 1),
            BottleNeck(128, 64),
            BottleNeck(128, 64),
            BottleNeck(128, 64),
            UpsampleBlock(128, 64),
            BottleNeck(64, 64),
            BottleNeck(64, 64),
            BottleNeck(64, 64),
            UpsampleBlock(64, 16),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(NUM_FEAT2, 256, kernel_size = 1),
            BottleNeck(256, 256),
            BottleNeck(256, 256),
            UpsampleBlock(256, 128),
            BottleNeck(128, 128),
            BottleNeck(128, 128),
            BottleNeck(128, 128),
            UpsampleBlock(128, 64),
            BottleNeck(64, 64),
            BottleNeck(64, 64),
            UpsampleBlock(64, 16),
            
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(NUM_FEAT3, 512, kernel_size = 1),
            BottleNeck(512, 512),
            BottleNeck(512, 512),
            BottleNeck(512, 512),
            UpsampleBlock(512, 256),
            BottleNeck(256, 256),
            BottleNeck(256, 256),
            BottleNeck(256, 256),
            UpsampleBlock(256, 128),
            UpsampleBlock(128, 64),
            BottleNeck(64, 64),
            BottleNeck(64, 64),
            UpsampleBlock(64, 16)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(NUM_FEAT4, 512, kernel_size = 1),
            UpsampleBlock(512, 512),
            BottleNeck(512, 512),
            BottleNeck(512, 512),
            BottleNeck(512, 512),
            UpsampleBlock(512, 256),
            BottleNeck(256, 256),
            BottleNeck(256, 256),
            BottleNeck(256, 256),
            UpsampleBlock(256, 128),
            UpsampleBlock(128, 64),
            UpsampleBlock(64, 16) 
        )

        self.conv1 = nn.Conv2d(64, 32, kernel_size = 1, padding = 0, stride = 1, bias = False)
        self.RGB = nn.Conv2d(32, 3, kernel_size=1,bias=False)
        
    def forward(self, res1, res2, res3, res4):
        
        res1 = self.branch1(res1)
        res2 = self.branch2(res2)
        res3 = self.branch3(res3)
        res4 = self.branch4(res4)
        x = torch.cat((res1, res2, res3, res4), 1)
        #x = res1 + res2 + res3 + res4
        #x = self.combine(x)
        x = self.relu(self.conv1(x))
        x = F.tanh(self.RGB(x)) / 2
        return x

if __name__ == '__main__':

    encoder = EncoderCell()

    decoder = DecoderCell()

    print(encoder)

    print(decoder)

    print('encoder_branch1 ', len(encoder.branch1.parameters()))
    print('encoder_branch2 ', len(encoder.branch2.parameters()))
    print('encoder_branch3 ', len(encoder.branch3.parameters()))
    print('encoder_branch4 ', len(encoder.branch4.parameters()))

    print('decoder_branch1 ', len(decoder.branch1.parameters()))
    print('decoder_branch2 ', len(decoder.branch2.parameters()))
    print('decoder_branch3 ', len(decoder.branch3.parameters()))
    print('decoder_branch4 ', len(decoder.branch4.parameters()))


