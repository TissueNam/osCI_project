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
from networks import build_wgan_gp_exact_network, get_optimizers_wgan, save_networks
from training import train_wgan_gp
from options import load_options, get_current_epoch
from comparison import evaluate_with_comparison

# from torch.utils.tensorboard import SummaryWriter
from tensorboard_logger import Logger

print("step 0. Set train options")
options = load_options(options)
dataloader = FlexibleCustomDataloader(fold='train', **options)
eval_dataloader = CustomDataloader(fold='test', **options)

networks = build_wgan_gp_exact_network(dataloader.num_classes, **options)
optimizers = get_optimizers_wgan(networks, finetune=True, **options)

# # for tensorboard plotting
# fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
# writer_real = SummaryWriter(f"logs/CIFAR_10/real")
# writer_fake = SummaryWriter(f"logs/CIFAR_10/fake")
# step = 0

print("step 1. train_wgan.py 시작")
start_epoch = get_current_epoch(options['result_dir']) + 1

# Set the logger
logger = Logger('./logs')
logger.writer.flush()
number_of_images = 10

for epoch in range(start_epoch, start_epoch + options['epochs']):
    print("train_wgan의 epoch: start_epoch:{}, epoch:{}/{}, 설정 eopch options['epochs']:{}"\
        .format(start_epoch, epoch, start_epoch + options['epochs']-1, options['epochs']))
    train_wgan_gp(networks, optimizers, dataloader, number_of_images, logger, epoch=epoch, total_epoch=(start_epoch + options['epochs']),**options)
    # eval_results = evaluate_with_comparison(networks, eval_dataloader, **options)
    # pprint(eval_results) # 분류기 학습시에만
    # 
    save_networks(networks, epoch, options['result_dir'])
