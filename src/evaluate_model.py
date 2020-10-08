import torch
from models.smc.src.crowd_count import CrowdCounter
from models.smc.src import network
import numpy as np


def evaluate_model(trained_model, data_loader, is_cuda=False):
    dtype = torch.FloatTensor if not is_cuda else torch.cuda.FloatTensor
    net = CrowdCounter()
    network.load_net(trained_model, net, dtype=dtype)
    net.eval()
    mae = 0.0
    mse = 0.0

    for blob in data_loader:                        
        im_data = blob['data']
        gt_data = blob['gt_density']
        density_map = net(im_data, gt_data)
        density_map = density_map.data.cpu().numpy()
        gt_count = np.sum(gt_data)
        et_count = np.sum(density_map)
        mae += abs(gt_count-et_count)
        mse += ((gt_count-et_count)*(gt_count-et_count))
        
    mae = mae/data_loader.get_num_samples()
    mse = np.sqrt(mse/data_loader.get_num_samples())
    return mae,mse