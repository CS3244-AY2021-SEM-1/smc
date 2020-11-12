import torch
import torch.nn as nn
from models.smc.src.network import Conv2d

class SMC(nn.Module):
    '''
    Switch Multi-column CSR CNN
    '''
    def __init__(self, bn=False, vary=False):
        super(SMC, self).__init__()
        
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

            # nn.MaxPool2d(2, stride=2),

            Conv2d(256, 512, 3, padding='same', bn=bn),
            Conv2d(512, 512, 3, padding='same', bn=bn),
            Conv2d(512, 512, 3, padding='same', bn=bn),

            # backend - fully conv layers (CSRNet backend B configuration)
            Conv2d(512, 512, 3, padding=2, bn=bn, dilation=2),
            Conv2d(512, 512, 3, padding=2, bn=bn, dilation=2),
            Conv2d(512, 512, 3, padding=2, bn=bn, dilation=2),
            Conv2d(512, 256, 3, padding=2, bn=bn, dilation=2),
            Conv2d(256, 128, 3, padding=2, bn=bn, dilation=2),
            Conv2d(128,  64, 3, padding=2, bn=bn, dilation=2)
        )        

        #self.fuse = (
        #    nn.Sequential(Conv2d(82,  1, 1, padding='same', bn=bn)) 
        #    if not vary 
        #    else nn.Sequential(Conv2d(84,  1, 1, padding='same', bn=bn))
        #)
        
        self.fuse1 = nn.Sequential(Conv2d(8, 1, 1, padding='same', bn=bn))
        self.fuse2 = nn.Sequential(Conv2d(10, 1, 1, padding='same', bn=bn))
        self.fuse3 = nn.Sequential(Conv2d(64, 1, 1, padding='same', bn=bn))
        
        self.final = nn.Sequential(Conv2d(1, 1, 1, padding='same', bn=bn))


    def forward(self, im_data, vary=False):
        
        # original shape: 1 x 3 x H x W
        
        # permuting to get to (1 x H x W x 3)
        permuted = im_data[0].permute(1,2,0)
        
        H = permuted.shape[0]
        top = permuted[:int(H/3)].permute(2,0,1).unsqueeze(0)
        mid = permuted[int(H/3):2*int(H/3)].permute(2,0,1).unsqueeze(0)
        btm = permuted[2*int(H/3):3*int(H/3)].permute(2,0,1).unsqueeze(0)
        
        r1 = self.r1(top)
        r2 = self.r2_1(mid)
        r3 = self.r3(btm)
        
        # joining back
        #tensor_list = [r1, r2, r3]
        #stacked_tensor = torch.stack(tensor_list).unsqueeze(0)        
        #print(stacked_tensor.shape)
        
        out1 = self.fuse1(r1)[0][0]
        out2 = self.fuse2(r2)[0][0]
        out3 = self.fuse3(r3)[0][0]
        
        #x = torch.cat((r1, r2, r3),1)
        #x = self.fuse(x)
        x = torch.cat((out1, out2, out3), 0).unsqueeze(0).unsqueeze(0)
        return x #self.final(x)

