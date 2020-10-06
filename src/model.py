import torch
import torch.nn as nn
from src.network import Conv2d

class SMC(nn.Module):
    '''
    Switch Multi-column CSR CNN
    '''

    def __init__(self, bn=False, vary=False):
        super(SMC self).__init__()
        
        # Foreground
        self.r1 = nn.Sequential(
            Conv2d(3, 16, 9, padding='same', bn=False),
            nn.MaxPool2d(2, stride=2),
            Conv2d(16, 32, 7, padding='same', bn=bn),
            nn.MaxPool2d(2, stride=2),
            Conv2d(32, 16, 7, padding='same', bn=bn),
            Conv2d(16,  8, 7, padding='same', bn=bn),
        )


        # Middleground
        self.r2_1 = nn.Sequential(
            Conv2d( 3, 20, 7, padding='same', bn=bn),
            nn.MaxPool2d(2, stride=2),
            Conv2d(20, 40, 5, padding='same', bn=bn),
            nn.MaxPool2d(2, stride=2),
            Conv2d(40, 20, 5, padding='same', bn=bn),
            Conv2d(20, 10, 5, padding='same', bn=bn)
        )

        self.r2_2 = nn.Sequential(
            Conv2d( 3, 24, 5, padding='same', bn=bn),
            nn.MaxPool2d(2, stride=2),
            Conv2d(24, 48, 3, padding='same', bn=bn),
            nn.MaxPool2d(2, stride=2),
            Conv2d(48, 24, 3, padding='same', bn=bn),
            Conv2d(24, 12, 3, padding='same', bn=bn)
        )

        # CSRNet Regressor - Background
        self.r3 = nn.Sequential(
            Conv2d(  3,  64, 3, padding='same', bn=bn), 
            Conv2d( 64,  64, 3, padding='same', bn=bn),
                                     
            nn.MaxPool2d(2, stride=2),
                                     
            Conv2d( 64, 128, 3, padding='same', bn=bn), 
            Conv2d(128, 128, 3, padding='same', bn=bn),
                                     
            nn.MaxPool2d(2, stride=2),
                
            Conv2d(128, 256, 3, padding='same', bn=bn), 
            Conv2d(256, 256, 3, padding='same', bn=bn),
            Conv2d(256, 256, 3, padding='same', bn=bn),
                
            nn.MaxPool2d(2, stride=2),
                
            Conv2d(256, 512, 3, padding='same', bn=bn),
            Conv2d(512, 512, 3, padding='same', bn=bn),
            Conv2d(512, 512, 3, padding='same', bn=bn),
                
            # backend - fully conv layers (CSRNet backend B configuration)
            Conv2d(512, 512, 3, padding=2, bn=bn, dilation=2),
            Conv2d(512, 512, 3, padding=2, bn=bn, dilation=2),
            Conv2d(512, 512, 3, padding=2, bn=bn, dilation=2),
            Conv2d(512, 256, 3, padding=2, bn=bn, dilation=2),
            Conv2d(256, 128, 3, padding=2, bn=bn, dilation=2),
            Conv2d(128,  64, 3, padding=2, bn=bn, dilation=2),
                
            # output layer
            Conv2d(64, 1, 1, padding='same', bn=bn)
        )        

        self.fuse    = (
            nn.Sequential(Conv2d(82,  1, 1, padding='same', bn=bn)) 
            if not vary 
            else nn.Sequential(Conv2d(84,  1, 1, padding='same', bn=bn))
        )


    def forward(self, im_data, vary=False):
        r1 = self.r1(im_data)
        r2 = self.r2_1(im_data) if not vary else self.r2_2(im_data)
        r3 = self.r3(im_data)
        x = torch.cat((r1, r2, r3), 1)
        x = self.fuse(x)
        return x
