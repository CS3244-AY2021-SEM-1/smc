import torch.nn as nn
from models.smc.src import network
from models.smc.src.model import SMC


class CrowdCounter(nn.Module):
    def __init__(self, is_cuda=False):
        super(CrowdCounter, self).__init__()        
        self.model = SMC(vary=False)
        self.criterion = nn.SmoothL1Loss()        
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
        density_map = nn.functional.interpolate(
            density_map, 
            (gt_data.shape[2], gt_data.shape[3]), 
            mode='bilinear',
            align_corners=True
        )
        
        
        if self.training:                        
            gt_data = network.np_to_variable(
                gt_data, 
                is_cuda=self.is_cuda, 
                is_training=self.training
            )

            self.loss_value = self.build_loss(density_map, gt_data)            
    
        return density_map
    
    def build_loss(self, density_map, gt_data):
        loss = self.criterion(density_map, gt_data)        
        return loss
    
    def get_model(self):
        return self.model