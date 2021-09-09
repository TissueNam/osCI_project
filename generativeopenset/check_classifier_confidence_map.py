#!/usr/bin/env python
import argparse
import os
import sys
from pprint import pprint


parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', help='Output directory for images and model checkpoints [default: .]', default='.')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for [default: 10]')
options = vars(parser.parse_args())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader, FlexibleCustomDataloader
from networks import build_VGG_basic_network, get_optimizers_basic
from options import load_options, get_current_epoch
from evaluation import draw_confidence_map, draw_basic_confidence_map

options = load_options(options)
dataloader = FlexibleCustomDataloader(fold='train', **options)
eval_dataloader = CustomDataloader(last_batch=True, shuffle=False, fold='test', **options)

networks = build_VGG_basic_network(dataloader.num_classes, **options)
optimizers = get_optimizers_basic(networks, finetune=False, **options)

start_epoch = get_current_epoch(options['result_dir']) + 1

new_results = draw_basic_confidence_map(networks, eval_dataloader, **options)
