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

# np.savez("./CS230_waferMap32_train.npz", x_train=nx_train, y_train=y_train_temp)
# np.savez("./CS230_waferMap32_test.npz", x_test=nx_test, y_test=y_test)

DATA_DIR = '/home/kcm/data'
DATASET_NAME = 'wafermap32'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)

WAFER_CLASSES = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full','none']

def main():
    print("step 1. Make the image folders")
    os.mkdir('wafermap32') if not path.exists('wafermap32') else print(f'The folder({DATASET_NAME}) already exists.')
    os.chdir(DATASET_PATH)
    
    print("Loading {} dataset files to {}...".format(DATASET_NAME, DATASET_PATH))
    trainLoader = np.load('./CS230_waferMap32_train.npz') # (15063, 32, 32, 3) (15063, 1)
    testLoader = np.load('./CS230_waferMap32_test.npz') # (3766, 32, 32, 3) (3766, 1)
    
    train_labels = trainLoader['y_train']
    train_data = trainLoader['x_train']
    train_filenames = []
    for lab in enumerate(train_labels):
        imgName = 'waferMap32_train_'+WAFER_CLASSES[int(lab[1][0])]+'_'+str(lab[0])+'.png'
        train_filenames.extend([imgName])
    
    test_labels = testLoader['y_test']
    test_data = testLoader['x_test']
    test_filenames = []
    for lab in enumerate(test_labels):
        imgName = 'waferMap32_test_'+WAFER_CLASSES[int(lab[1][0])]+'_'+str(lab[0])+'.png'
        test_filenames.extend([imgName])   

    examples = []
    for lab, fn, dat in tqdm(zip(train_labels, train_filenames, train_data)):
        example = make_example(lab, fn, dat)
        example['fold'] = 'train'
        examples.append(example)
        
    for lab, fn, dat in tqdm(zip(test_labels, test_filenames, test_data)):
        example = make_example(lab, fn, dat)
        example['fold'] = 'test'
        examples.append(example)
        
    print("Saving .dataset files...")
    output_filename = '{}/{}.dataset'.format(DATA_DIR, DATASET_NAME)
    save_image_dataset(examples, output_filename)
    
#     splits = [
#         [3, 6, 7, 8],
#         [1, 2, 4, 6],
#         [2, 3, 4, 9],
#         [0, 1, 2, 6],
#         [4, 5, 6, 9],
#     ]

    splits = [
        [3, 6, 7, 8],
        [1, 2, 4, 6],
        [2, 3, 4, 7],
        [0, 1, 2, 6],
        [4, 5, 6, 8],
    ]
    
    # For MORE open set classification experiments
    cnt = 0
    for class_idx in ['none']:
        fp_set_name = '{}/{}-{}.dataset'.format(DATA_DIR, DATASET_NAME, class_idx)
        fp_openset_name = '{}/{}-not{}.dataset'.format(DATA_DIR, DATASET_NAME, class_idx)
        fp_set = open(fp_set_name, 'w')
        fp_openset = open(fp_openset_name, 'w')
        for example in examples:
            if example['label'] == class_idx:
                fp_set.write(json.dumps(example) + '\n')
                cnt += 1
            else:
                fp_openset.write(json.dumps(example) + '\n')
        fp_set.close()
        fp_openset.close()
    print("Wrote {} items to {}".format(cnt, fp_set_name))
    print("Wrote {} items to {}".format(len(examples)-cnt, fp_openset_name))

    for idx, split in enumerate(splits):
        unknown_classes = [waferMap32_class(i) for i in split]
        known_examples = [e for e in examples if e['label'] not in unknown_classes]
        unknown_examples = [e for e in examples if e['label'] in unknown_classes]
        known_filename = '{}/{}-split{}a.dataset'.format(DATA_DIR, DATASET_NAME, idx)
        unknown_filename = '{}/{}-split{}b.dataset'.format(DATA_DIR, DATASET_NAME, idx)
        save_image_dataset(known_examples, known_filename)
        save_image_dataset(unknown_examples, unknown_filename)
    print("Finished writing datasets")
    
def save_image_dataset(examples, output_filename):
    fp = open(output_filename, 'w')
    for line in examples:
        fp.write(json.dumps(line) + '\n')
    fp.close()
    print("Wrote {} items to {}".format(len(examples), output_filename))
        
def make_example(label, filename, data):
    pixels = data*255
    pixels = pixels.astype(np.uint8)
    filename = filename
    Image.fromarray(pixels).save(filename)
    class_name = waferMap32_class(int(label[0]))
    return {
            'filename': os.path.join(DATASET_PATH, filename),
            'label': class_name,
    }

def waferMap32_class(label_idx):
    return WAFER_CLASSES[label_idx]
        
if __name__ == "__main__":
    main()