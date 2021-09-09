#!/usr/bin/env python
import argparse
import os
import sys
from pprint import pprint

# Print --help message before importing the rest of the project
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')
parser.add_argument('--fold', default="test", help='Name of evaluation fold [default: test]')
parser.add_argument('--epochs', type=int, default=845, help='number of epochs to train for [default: 10]')
# parser.add_argument('--epoch', type=int, help='Epoch to evaluate (latest epoch if none chosen)')
parser.add_argument('--gen_epoch', type=int, default=200000, help='epoch of trained WGAN-GP epoch [defualt: 1000]')
parser.add_argument('--comparison_dataset', type=str, help='Dataset for off-manifold comparison')
parser.add_argument('--aux_dataset', type=str, help='aux_dataset used in train_classifier')
parser.add_argument('--mode', default='', help='One of: default, weibull, weibull-kplus1, baseline')
parser.add_argument('--roc_output', type=str, help='Optional filename for ROC data output')
options = vars(parser.parse_args())

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from dataloader import CustomDataloader
# from networks import build_networks
# from options import load_options, get_current_epoch
# from evaluation import save_evaluation
# from comparison import evaluate_with_comparison
from dataloader import CustomDataloader, FlexibleCustomDataloader
from networks import build_VGG_semi_sup_network_multibranch, get_optimizers_semisuper, save_classification_networks
from options import load_options, get_current_epoch
from comparison import semi_supervised_evaluate_with_comparison
from evaluation import save_evaluation
from training import train_wresnet_classifier, train_wresnet_classifier_multibranch


options = load_options(options)
# if not options.get('epoch'):
#     options['epoch'] = get_current_epoch(options['result_dir'])
# TODO: Globally disable dataset augmentation during evaluation
# 옵션 epoch: 마지막까지 돌아간 횟수(ex. 34)
options['random_horizontal_flip'] = False

dataloader = CustomDataloader(last_batch=True, shuffle=False, **options) # dataset의 test data

# TODO: structure options in a way that doesn't require this sort of hack
train_dataloader_options = options.copy()
train_dataloader_options['fold'] = 'train'
dataloader_train = CustomDataloader(last_batch=True, shuffle=False, **train_dataloader_options) # dataset의 train data


networks = build_VGG_semi_sup_network_multibranch(dataloader.num_classes, epoch=options['epochs'], **options)
new_results = semi_supervised_evaluate_with_comparison(networks, dataloader, dataloader_train=dataloader_train, **options)

pprint(new_results)

save_evaluation(new_results, options['result_dir'], options['epochs'])
