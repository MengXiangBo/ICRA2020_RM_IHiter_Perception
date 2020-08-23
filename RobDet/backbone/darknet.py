import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class DarkNet_Tiny(nn.Module):
    def __init__(self, num_classes=1000):
        
        super(DarkNet_Tiny, self).__init__()
        # backbone network : DarkNet-Tiny
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            Conv_BN_LeakyReLU(32, 32, 3, padding=1, stride=2)
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            Conv_BN_LeakyReLU(64, 64, 3, padding=1, stride=2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 128, 3, padding=1, stride=2),
        )

        # output : stride = 16, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 256, 3, padding=1, stride=2),
        )

        # output : stride = 32, c = 512
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 512, 3, padding=1, stride=2),
        )

        # self.conv_6 = nn.Conv2d(512, 1000, 1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        C_3 = self.conv_3(x)
        C_4 = self.conv_4(C_3)
        C_5 = self.conv_5(C_4)
        # x = self.conv_6(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        return C_4, C_5


class DarkNet_Light(nn.Module):
    def __init__(self, num_classes=1000):
        
        super(DarkNet_Light, self).__init__()
        # backbone network : DarkNet_Light
        self.conv_1 = Conv_BN_LeakyReLU(3, 16, 3, 1)
        self.maxpool_1 = nn.MaxPool2d((2, 2), 2)              # stride = 2

        self.conv_2 = Conv_BN_LeakyReLU(16, 32, 3, 1)
        self.maxpool_2 = nn.MaxPool2d((2, 2), 2)              # stride = 4

        self.conv_3 = Conv_BN_LeakyReLU(32, 64, 3, 1)
        self.maxpool_3 = nn.MaxPool2d((2, 2), 2)              # stride = 8

        self.conv_4 = Conv_BN_LeakyReLU(64, 128, 3, 1)
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)              # stride = 16

        self.conv_5 = Conv_BN_LeakyReLU(128, 256, 3, 1)
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)              # stride = 32

        self.conv_6 = Conv_BN_LeakyReLU(256, 512, 3, 1)
        self.maxpool_6 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d((2, 2), 1)                           # stride = 32
        )

        self.conv_7 = Conv_BN_LeakyReLU(512, 1024, 3, 1)


    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.maxpool_2(x)
        x = self.conv_3(x)
        x = self.maxpool_3(x)
        x = self.conv_4(x)
        x = self.maxpool_4(x)
        C_4 = self.conv_5(x)       # stride = 16
        x = self.maxpool_5(C_4)  
        x = self.conv_6(x)
        x = self.maxpool_6(x)
        C_5 = self.conv_7(x)     # stride = 32

        return C_4, C_5


def darknet_tiny(pretrained=False, hr=False, **kwargs):
    """Constructs a darknet-tiny model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet_Tiny()
    if pretrained:
        print('Loading the pretrained model ...')
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        if hr:
            print('Loading the hi-res darknet_tiny-448 ...')
            model.load_state_dict(torch.load(path_to_dir + '/weights/darknet_tiny/darknet_tiny_hr_61.85.pth', map_location='cuda'), strict=False)
        else:
            print('Loading the darknet_tiny ...')
            model.load_state_dict(torch.load(path_to_dir + '/weights/darknet_tiny/darknet_tiny_63.50_85.06.pth', map_location='cuda'), strict=False)
    return model


def darknet_light(pretrained=False, hr=False, **kwargs):
    """Constructs a darknet_light model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet_Light()
    if pretrained:
        print('Loading the pretrained model ...')
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        if hr:
            print('Loading the hi-res darknet_light-448 ...')
            model.load_state_dict(torch.load(path_to_dir + '/weights/darknet_light/darknet_light_hr_59.61.pth', map_location='cuda'), strict=False)
        else:
            print('Loading the darknet_light ...')
            model.load_state_dict(torch.load(path_to_dir + '/weights/darknet_light/darknet_light_58.99.pth', map_location='cuda'), strict=False)
    return model
