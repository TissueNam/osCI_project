#!/usr/bin/env python
import numpy as np
import hashlib
import sys
import requests
import os
import os.path
from os import path
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm


DATA_DIR = '/home/kcm/data'
DATASET_NAME = 'wafermap32'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)

def download_wafermap32():
    print("\nstep 2. load the ''LSWMD.pkl''file")
    (train_x, train_y), (test_x, test_y) = download_wafermap32_data()

    print("\nstep 3. Writing wafermap32 dataset")
    train = save_set('train', train_x, train_y)
    test = save_set('test', test_x, test_y)
    
    for example in train:
        example['fold'] = 'train'
    for example in test:
        example['fold'] = 'test'
    with open('wafermap32.dataset', 'w') as fp:
        for example in train + test:
            fp.write(json.dump(example, sort_key=True) + '\n')
            
    # For open set classification exeriments
    with open('wafermap32_withPattern.dataset', 'w') as fp:
        for example in train + test:
            if int(example['label']) <= 7:
                fp.write(json.dumps(example) + '\n')
    with open('wafermap32_nonPattern.dataset', 'w') as fp:
        for example in train + test:
            if int(example['label']) > 7:
                fp.write(json.dumps(example) + '\n')                
        
    # For More open set classification experiments
    for class_idx in range(9):
        fp_set = open('wafermap32-{}.dataset'.format(class_idx), 'w')
        fp_openset = open('wafermap32-not{}.dataset'.format(class_idx), 'w')
        for example in train + test:
            if int(example['label']) == class_idx:
                fp_set.write(json.dump(example) + '\n')
            else:
                fp_openset.write(json.dump(example) + '\n')
        fp_set.close()
        fp_openset.close()
            
    # For experiments varying openness
    for class_idx in [2, 3, 4, 5, 6, 7]:
        fp_set = open('wafermap32-0{}.dataset'.format(class_idx), 'w')
        fp_openset = open('wafermap32-{}9.dataset'.format(class_idx), 'w')
        for example in train+test:
            if int(example['label']) <= class_idx:
                fp_set.write(json.dumps(example) + '\n')
            else:
                fp_openset.write(json.dumps(example) + '\n')
        fp_set.close()
        fp_openset.close()
        
    # Splits to match the CIFAR and SVHN experiments
    splits = [
        [3, 6, 7, 8],
        [1, 2, 4, 6],
        [2, 3, 4, 9],
        [0, 1, 2, 6],
        [4, 5, 6, 9],
    ]
    examples = train + test
    for idx, split in enumerate(splits):
        known_examples = [e for e in examples if int(e['label']) not in split]
        unknown_examples = [e for e in examples if int(e['label']) in split]
        known_filename = '{}/{}-split{}a.dataset'.format(DATA_DIR, DATASET_NAME, idx)
        unknown_filename = '{}/{}-split{}b.dataset'.format(DATA_DIR, DATASET_NAME, idx)
        save_image_dataset(known_filename, known_examples)
        save_image_dataset(unknown_filename, unknown_examples)
        
def save_image_dataset(filename, examples):
    with open(filename, 'w') as fp:
        for ex in examples:
            fp.write(json.dump(ex))
            fp.write('\n')
        
def download_wafermap32_data(path='wafermap32.npz'):
    print("download_wafermap32_data")
    with np.load(path) as f:
        x_test, y_test = f['x_test'], f['y_test']
        x_train, y_train = f['x_train'], f['y_train']
    return (x_train, y_train), (x_test, y_test)


def save_set(fold, x, y, suffix='png'):
    example = []
    fp = open('wafermap32_{}.dataset'.format(fold), 'w')
    print("Writing wafermap32 dataset {}".format(fold))
    for i in tqdm(range(len(x))):
        k = 1
    fp.close()
    return example


if __name__ == "__main__":
    os.chdir(DATA_DIR)
    print("step 1. Make the image folders")
    os.mkdir('wafermap32') if not path.exists('wafermap32') else print('The folder(''wafermap32'') already exists.')
