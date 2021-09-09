#!/usr/bin/env python
import argparse
import os
import sys
from pprint import pprint

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
DEFECT_CLASSES = ['Array', 'Gap', 'Particle', 'Scratch', 'Undefined']
save_path = './failed_check_fold'

def failed_data_check(networks, dataloader, **options):
    for net in networks.values():
        net.eval()

    netC = networks['classifier']    
    fold = options.get('fold', 'evaluation')

    classification_closed_correct = 0
    classification_total = 0

    cnt = 0
    with torch.no_grad():
        for i, (images, class_labels) in enumerate(dataloader):
            images = images.unsqueeze(1)
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(class_labels, requires_grad=False).cuda()
            # cur_batch_size = images.shape[0]

            net_y = netC(images)
            class_predictions = F.softmax(net_y, dim=1)

            # 결과 조사, 확률 조사 정답 가능성

            _, predicted = class_predictions.max(1)
            for idx, pred in enumerate(predicted.data):
                if (pred != labels[idx]):
                    cnt += 1
                    print(DEFECT_CLASSES[labels[idx]], DEFECT_CLASSES[pred])
                    file_name = save_path + "/true_" + DEFECT_CLASSES[labels[idx]] + "_to_" + DEFECT_CLASSES[pred] + "_" + str(cnt) +".png"
                    data = images[idx].cpu().numpy()
                    data = (data-data.min())/(data.max()-data.min())
                    data = data*255
                    pixels = data.reshape((512, 512)).astype(np.uint8)
                    Image.fromarray(pixels).save(file_name)
            
            classification_closed_correct += sum(predicted.data == labels)
            classification_total += len(labels)

    print(cnt)
    basic_stat = {
        fold: {
            'closed_set_image_class_accuracy': float(classification_closed_correct) / (classification_total),
        }
    }


    return basic_stat

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

# start_epoch = get_current_epoch(options['result_dir']) + 1

evalate = failed_data_check(networks, eval_dataloader, **options)

print(evalate)