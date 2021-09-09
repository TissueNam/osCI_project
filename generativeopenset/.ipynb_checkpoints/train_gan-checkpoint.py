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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader, FlexibleCustomDataloader
from training import train_gan
from networks import build_networks, save_networks, get_optimizers
from options import load_options, get_current_epoch
from counterfactual import generate_counterfactual
from comparison import evaluate_with_comparison

print("step 0. train_gan.py 실행")
options = load_options(options)
dataloader = FlexibleCustomDataloader(fold='train', **options)
eval_dataloader = CustomDataloader(fold='test', **options)

networks = build_networks(dataloader.num_classes, **options)
optimizers = get_optimizers(networks, **options)

print("step 1. train_gan.py 시작")
start_epoch = get_current_epoch(options['result_dir']) + 1
for epoch in range(start_epoch, start_epoch + options['epochs']):
    print("train_gan의 epoch: start_epoch:{}, epoch:{}/{}, 설정 eopch options['epochs']:{}".format(start_epoch, epoch, start_epoch + options['epochs'], options['epochs']))
    print("train_gan 학습! {}".format(epoch))
    train_gan(networks, optimizers, dataloader, epoch=epoch, **options)
    #generate_counterfactual(networks, dataloader, **options)
    print("evaluate_with_comparison 실행! {}".format(epoch))
    eval_results = evaluate_with_comparison(networks, eval_dataloader, **options)
    pprint(eval_results)
    save_networks(networks, epoch, options['result_dir'])
