import argparse
import os
import sys
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', help='Output directory for images and model checkpoints [default: .]', default='.')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train for [default: 10]')
parser.add_argument('--gen_epoch', type=int, default=200000, help='epoch of trained WGAN-GP epoch [defualt: 1000]')
parser.add_argument('--alpha1', type=float, default=0.8, help='classifier loss offset(default=0.8)')
parser.add_argument('--alpha2', type=float, default=0.2, help='semi-supervised loss offset(default=0.2)')
parser.add_argument('--transNum', type=int, default=14, help='how many generate transformed image with one concated image')
options = vars(parser.parse_args())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader, FlexibleCustomDataloader
from networks import build_VGG_semi_sup_network_multibranch, get_optimizers_semisuper, save_classification_networks
from options import load_options, get_current_epoch
from comparison import evaluate_model_accuracy, semi_supervised_evaluate_with_comparison, classifier_evaluate_with_comparison

# from torch.utils.tensorboard import SummaryWriter
from tensorboard_logger import Logger

options = load_options(options)
dataloader = FlexibleCustomDataloader(fold='train', **options)
eval_dataloader = CustomDataloader(last_batch=True, shuffle=False, fold='train', **options)
# eval_dataloader = FlexibleCustomDataloader(last_batch=True, shuffle=False, fold='test', **options)

# networks = build_semi_sup_network(dataloader.num_classes, **options)
# optimizers = get_optimizers_wgan(networks, finetune=True, **options)

networks = build_VGG_semi_sup_network_multibranch(dataloader.num_classes, epoch=options['epochs'], **options)
optimizers = get_optimizers_semisuper(networks, finetune=False, **options)

# Set the logger
logger = Logger('./logs')
logger.writer.flush()
number_of_images = 10

start_epoch = get_current_epoch(options['result_dir']) + 1

# eval_results = semi_supervised_evaluate_with_comparison(networks, eval_dataloader, **options)
# pprint(eval_results)

# eval_results_SS = semi_supervised_SS_evaluate_with_comparison(networks, eval_dataloader, **options)
# pprint(eval_results_SS)

# eval_results, eval_results_SS = evaluate_model_accuracy(networks, eval_dataloader, **options)
eval_results = classifier_evaluate_with_comparison(networks, eval_dataloader, **options)
pprint(eval_results)
# pprint(eval_results_SS)