from torchvision import models
from torchsummary import summary
from networks import build_networks

#!/usr/bin/env python
import argparse
import os
import sys
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', help='Output directory for images and model checkpoints [default: .]', default='.')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for [default: 10]')
parser.add_argument('--aux_dataset', help='Path to aux_dataset file [default: None]')

options = vars(parser.parse_args())
print(options)


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader, FlexibleCustomDataloader
from training import train_gan
from networks import build_networks, save_networks, get_optimizers
from options import load_options, get_current_epoch
from counterfactual import generate_counterfactual
from comparison import evaluate_with_comparison

import torch
from torchviz import make_dot


options = load_options(options)
dataloader = FlexibleCustomDataloader(fold='train', **options)
eval_dataloader = CustomDataloader(fold='test', **options)

networks = build_networks(dataloader.num_classes, **options)
optimizers = get_optimizers(networks, **options)

summary(networks, (1, 3, 224, 224), device='cpu')

#print(networks)