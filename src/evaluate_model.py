import torch
from models.smc.src.crowd_count import CrowdCounter
from models.smc.src import network
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def analyse_loader(loader, title='Breakdown of the images in the loader by type'):
    d = {'High': 0, 'Med': 0, 'Low': 0, 'None': 0, 'Fog': 0, 'Rain': 0, 'Snow': 0}
    colours = ['orange', 'orange', 'orange', 'green', 'green', 'green', 'green']
    labels = ['crowd', 'weather', 'weather', 'weather', 'weather', 'weather', 'weather']
    for id, blob in enumerate(loader):
        metadata = blob['metadata']
        d[metadata['crowd_density']] += 1
        d[metadata['weather']] += 1
    for i, key in enumerate(d):
        plt.text(i, d[key], '{}%'.format(d[key]*100/loader.get_num_samples()), 
                 ha='center', va='bottom', size=10)
    pd.Series(d).plot(kind='bar', figsize=(8,5), color=colours)
    leg1 = mpatches.Patch(color='orange', label='Crowd Density')
    leg2 = mpatches.Patch(color='green', label='Weather')
    plt.legend(handles=[leg1, leg2])
    plt.title(title)
    plt.show()
    

def evaluate_model(trained_model, data_loader, is_cuda=False):
    dtype = torch.FloatTensor if not is_cuda else torch.cuda.FloatTensor
    net = CrowdCounter(is_cuda=is_cuda)
    network.load_net(trained_model, net, dtype=dtype)
    
    if is_cuda:
        net.cuda()
    net.eval()
    
    # values
    MAEcrowddensity = {'High': 0, 'Med': 0, 'Low': 0}
    MSEcrowddensity = {'High': 0, 'Med': 0, 'Low': 0}
    MAPEcrowddensity = {'High': 0, 'Med': 0, 'Low': 0}
    MAEweather = {'None': 0, 'Fog': 0, 'Rain': 0, 'Snow': 0}
    MSEweather = {'None': 0, 'Fog': 0, 'Rain': 0, 'Snow': 0}
    MAPEweather = {'None': 0, 'Fog': 0, 'Rain': 0, 'Snow': 0}
    MAE = 0.0
    MSE = 0.0
    MAPE = 0.0

    # counts for averaging
    crowddensity_count = {'High': 0, 'Med': 0, 'Low': 0}
    weather_count = {'None': 0, 'Fog': 0, 'Rain': 0, 'Snow': 0}
    

    with torch.no_grad():
        for blob in data_loader:                        
            im_data = blob['data']
            gt_data = blob['gt_density']
            metadata = blob['metadata']

            crowd_density = metadata['crowd_density']
            weather = metadata['weather']

            density_map = net(im_data, gt_data)
            density_map = density_map.data.cpu().numpy()
            gt_count = np.sum(gt_data)
            et_count = np.sum(density_map)

            # updating the values
            MAEcrowddensity[crowd_density] += abs(gt_count-et_count)
            MSEcrowddensity[crowd_density] += ((gt_count-et_count)*(gt_count-et_count))
            MAPEcrowddensity[crowd_density] += abs((gt_count-et_count)/gt_count)
            MAEweather[weather] += abs(gt_count-et_count)
            MSEweather[weather] += ((gt_count-et_count)*(gt_count-et_count))
            MAPEweather[weather] += abs((gt_count-et_count)/gt_count)
            MAE += abs(gt_count-et_count)
            MSE += ((gt_count-et_count)*(gt_count-et_count))
            MAPE += abs((gt_count-et_count)/gt_count)

            # updating the counts
            crowddensity_count[crowd_density] += 1
            weather_count[weather] += 1

    # averaging
    for key in crowddensity_count:
        MAEcrowddensity[key] = MAEcrowddensity[key] / crowddensity_count[key] if crowddensity_count[key] else 0
        MSEcrowddensity[key] = MSEcrowddensity[key] / crowddensity_count[key] if crowddensity_count[key] else 0
        MAPEcrowddensity[key] = MAPEcrowddensity[key] / crowddensity_count[key] if crowddensity_count[key] else 0

    # averaging
    for key in weather_count:
        MAEweather[key] = MAEweather[key] / weather_count[key] if weather_count[key] else 0
        MSEweather[key] = MSEweather[key] / weather_count[key] if weather_count[key] else 0
        MAPEweather[key] = MAPEweather[key] / weather_count[key] if weather_count[key] else 0
    
    # averaging
    MAE /= data_loader.get_num_samples()
    MSE /= data_loader.get_num_samples()
    MAPE /= data_loader.get_num_samples()
    RMSE = np.sqrt(MSE)

    return MAEcrowddensity, MSEcrowddensity, MAPEcrowddensity, MAEweather, MSEweather, MAPEweather, MAE, MSE, RMSE, MAPE


