import torch.nn as nn
import torch
from models.smc.src import network
from models.smc.src.model import SMC
import numpy as np

class CrowdCounter(nn.Module):
    def __init__(self, is_cuda=False):
        super(CrowdCounter, self).__init__()        
        self.model = SMC(vary=False)
        self.criterion = nn.MSELoss(reduction='sum').cuda()    
        self.is_cuda=is_cuda
        
    @property
    def loss(self):
        return self.loss_value

    def forward(self, im_data, gt_data=None):        
        im_data = network.np_to_variable(
            im_data, 
            is_cuda=self.is_cuda, 
            is_training=self.training
        )

        # generating density map + upsampling to match the gt_data shape
        density_map = self.model(im_data)
        
        
        if self.training:
            
            gt_data = gt_data[0].permute(1,2,0)
            H = int(gt_data.shape[0]/3)
            gt_data = gt_data[:3*H].permute(2,0,1).unsqueeze(0)
            
            gt_data = network.np_to_variable(
                gt_data, 
                is_cuda=self.is_cuda, 
                is_training=self.training
            )
            gt_data = nn.functional.interpolate(gt_data, (density_map.shape[2], density_map.shape[3]), mode='bilinear', align_corners=True) * 16
            
            self.loss_value = self.build_loss(density_map, gt_data)
    
        return density_map
    
    def build_loss(self, density_map, gt_data):
        loss = self.criterion(density_map, gt_data)        
        return loss
    
    def get_model(self):
        return self.model