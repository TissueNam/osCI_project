# from SemiSupervisedOpenSet.SemiSupervisedOpenSet_CIFAR10_n1.generativeopenset.comparison import evaluate_with_comparison
import argparse
import os
import sys
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', help='Output directory for images and model checkpoints [default: .]', default='.')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for [default: 10]')
parser.add_argument('--gen_epoch', type=int, default=200000, help='epoch of trained WGAN-GP epoch [defualt: 1000]')
parser.add_argument('--alpha1', type=float, default=0.8, help='classifier loss offset(default=0.8)')
parser.add_argument('--alpha2', type=float, default=0.2, help='semi-supervised loss offset(default=0.2)')
parser.add_argument('--transNum', type=int, default=14, help='how many generate transformed image with one concated image')
options = vars(parser.parse_args())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader, FlexibleCustomDataloader
from networks import build_VGG_semi_sup_network_multibranch, get_optimizers_semisuper, get_lr_scheduler, save_classification_networks, save_networks
from options import load_options, get_current_epoch
from comparison import semi_supervised_evaluate_with_comparison, classifier_evaluate_with_comparison
from evaluation import save_evaluation
from training import train_wresnet_classifier, train_VGG_classifier_multibranch

# from torch.utils.tensorboard import SummaryWriter
from tensorboard_logger import Logger

options = load_options(options)
dataloader = FlexibleCustomDataloader(fold='train', **options)
eval_dataloader = CustomDataloader(last_batch=True, shuffle=False, fold='test', **options)

networks = build_VGG_semi_sup_network_multibranch(dataloader.num_classes, **options)
optimizers = get_optimizers_semisuper(networks, finetune=False, **options)
schedulers = get_lr_scheduler(networks, optimizers, **options)

# Set the logger
logger = Logger('./logs')
logger.writer.flush()
number_of_images = 10

# figure setting
epoch_list = np.asarray([*range(1, options['epochs']+1, 1)])
epoch_list += get_current_epoch(options['result_dir'])
accuracy_list = np.zeros(options['epochs'])
SS_accuracy_list = np.zeros(options['epochs'])
cnt = 0

start_epoch = get_current_epoch(options['result_dir']) + 1
for epoch in range(start_epoch, start_epoch + options['epochs']):
    print(f"Current epoch = {epoch}/{start_epoch+options['epochs']-1}")
    train_VGG_classifier_multibranch(networks, optimizers, schedulers, dataloader, epoch=epoch, **options)
    eval_results = classifier_evaluate_with_comparison(networks, eval_dataloader, **options)
    accuracy_list[cnt] = eval_results[0]['evaluation']['closed_set_image_class_accuracy']
    SS_accuracy_list[cnt] = eval_results[1]['evaluation']['closed_set_self_supervised_accuracy']
    cnt += 1

    pprint(eval_results)
    save_evaluation(eval_results, options['result_dir'], epoch)
    save_classification_networks(networks, epoch, options['result_dir'])

# save figure
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(epoch_list, accuracy_list, label='Accuracy(image class)')
ax.plot(epoch_list, SS_accuracy_list, label='Accuracy(transpose class)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
plt.title("Accuracy(transpose class)")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig("./210726_wafermap_accuracy.png")
print("End~")