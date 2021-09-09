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
print(DATASET_PATH)

WAFER_CLASSES = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full','none']

def set_dataset():
    print("Loading {} dataset files to {}...".format(DATASET_NAME, DATASET_PATH))
    print(os.getcwd())
    trainLoader = np.load('CS230_waferMap32_train.npz') # (15063, 32, 32, 3) (15063, 1)
    testLoader = np.load('./CS230_waferMap32_test.npz') # (3766, 32, 32, 3) (3766, 1)

    train_labels = trainLoader['y_train']
    train_data = trainLoader['x_train']

    print(train_data.shape)

    return True

if __name__ == "__main__":
    os.chdir(DATASET_PATH)
    print("step 1. Make the image folders")
    os.mkdir('wafermap32_custom') if not path.exists('wafermap32_custom') else print(f'The folder({DATASET_NAME}) already exists.')

    print("step 2. Setting Datasets")
    set_dataset()
