# from SemiSupervisedOpenSet.SemiSupervisedOpenSet_CIFAR10_animal_machine.generativeopenset.options import get_current_epoch
import argparse
import os
import sys
from pprint import pprint

import torch
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', help='Output directory for images and model checkpoints [default: .]', default='.')
options = vars(parser.parse_args())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader, FlexibleCustomDataloader
from options import load_options, get_current_epoch

options = load_options(options)
dataloader = FlexibleCustomDataloader(fold='train', **options)
eval_dataloader = CustomDataloader(last_batch=True, shuffle=False, fold='test', **options)

def get_mean_std(loader):
    # VAR[X] = E[X**2] - E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    return mean, std

mean, std = get_mean_std(dataloader)

print("train dataset mean= ",mean)
print("train dataset std= ",std)

# mean, std = get_mean_std(eval_dataloader)

# print("test dataset mean= ",mean)
# print("test dataset std= ",std)