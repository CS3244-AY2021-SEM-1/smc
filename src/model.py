import torch
import torch.nn as nn
from models.smc.src.network import Conv2d

class SMC(nn.Module):
    '''
    Switch Multi-column CSR CNN
    '''

    def __init__(self, bn=False, vary=False):
        super(SMC, self).__init__()
        
        self.branch1 = nn.Sequential(Conv2d( 3, 16, 9,bn=bn),
                                     Conv2d(16, 32, 7, bn=bn),
                                     Conv2d(32, 16, 7, bn=bn),
                                     Conv2d(16,  8, 7, bn=bn))
        
        self.branch2 = nn.Sequential(Conv2d( 3, 20, 7, bn=bn),
                                     Conv2d(20, 40, 5, bn=bn),
                                     Conv2d(40, 20, 5, bn=bn),
                                     Conv2d(20, 10, 5, bn=bn))
        
        self.branch3 = nn.Sequential(Conv2d( 3, 24, 5, bn=bn),
                                     Conv2d(24, 48, 3, bn=bn),
                                     Conv2d(48, 24, 3, bn=bn),
                                     Conv2d(24, 12, 3, bn=bn))
        
        self.fuse = nn.Sequential(Conv2d( 30, 1, 1, bn=bn))
        
    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1,x2,x3),1)
        x = self.fuse(x)
        
        return x
