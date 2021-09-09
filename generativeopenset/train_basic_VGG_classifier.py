import argparse
import os
import sys
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', help='Output directory for images and model checkpoints [default: .]', default='.')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for [default: 10]')
options = vars(parser.parse_args())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader, FlexibleCustomDataloader
from networks import build_VGG_basic_network, get_optimizers_basic, get_lr_scheduler, save_basic_classification_networks
from options import load_options, get_current_epoch
from comparison import classifier_evaluate_with_comparison_basic
from evaluation import save_evaluation
from training import train_VGG_basic_classifier

options = load_options(options)
dataloader = FlexibleCustomDataloader(fold='train', **options)
eval_dataloader = CustomDataloader(last_batch=True, shuffle=False, fold='test', **options)

networks = build_VGG_basic_network(dataloader.num_classes, **options)
optimizers = get_optimizers_basic(networks, finetune=False, **options)
schedulers = get_lr_scheduler(networks, optimizers, **options)

# figure setting
epoch_list = np.asarray([*range(1, options['epochs']+1, 1)])
epoch_list += get_current_epoch(options['result_dir'])
accuracy_list = np.zeros(options['epochs'])
# SS_accuracy_list = np.zeros(options['epochs'])
cnt = 0

start_epoch = get_current_epoch(options['result_dir']) + 1
for epoch in range(start_epoch, start_epoch + options['epochs']):
    print(f"Current epoch = {epoch}/{start_epoch+options['epochs']-1}")
    train_VGG_basic_classifier(networks, optimizers, schedulers, dataloader, epoch=epoch, **options)
    eval_results = classifier_evaluate_with_comparison_basic(networks, eval_dataloader, **options)
    accuracy_list[cnt] = eval_results['evaluation']['closed_set_image_class_accuracy']
    # SS_accuracy_list[cnt] = eval_results[1]['evaluation']['closed_set_self_supervised_accuracy']
    cnt += 1

    pprint(eval_results)
    save_evaluation(eval_results, options['result_dir'], epoch)
    save_basic_classification_networks(networks, epoch, options['result_dir'])

# save figure
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(epoch_list, accuracy_list, label='Accuracy(image class)')
# ax.plot(epoch_list, SS_accuracy_list, label='Accuracy(transpose class)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
plt.title("Accuracy(transpose class)")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig("./210812_waferDefect_accuracy.png")
print("End~")