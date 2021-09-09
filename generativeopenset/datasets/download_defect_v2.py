import numpy as np
import os
import os.path
from os import path
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm

DATA_DIR = '/home/kcm/data'
DATASET_NAME = 'waferDefect_v2'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)

# DEFECT_CLASSES = ['Array', 'Cone', 'LargeGap', 'Line', 'Particle', 'Pattern', 'Pattern_con', 'Residue', 'Scratch', 'SmallGap', 'where', 'Spot']
# DEFECT_CLASSES = ['Array', 'Cone', 'LargeGap', 'Line', 'Particle', 'Pattern', 'Pattern_con', 'Residue', 'Scratch', 'SmallGap', 'where']
# DEFECT_CLASSES = ['Array', 'Gap', 'Line', 'MicroScractch', 'Particle', 'Pattern', 'Pattern_con', 'Scratch', 'where', 'Cone', 'Spot']
DEFECT_CLASSES = ['Array', 'Gap', 'Line', 'MicroScractch', 'Particle', 'Pattern', 'Pattern_con', 'Scratch', 'where', 'Cone']
def main():
    print("step 1. Make the image folders")
    os.mkdir(DATASET_NAME) if not path.exists(DATASET_NAME) else print(f'The folder({DATASET_NAME}) already exists.')
    os.chdir(DATASET_PATH)

    print(f"Loading {DATASET_NAME} dataset files to {DATASET_PATH}...")
    print(os.getcwd())
    train_data = np.load('./Defect_train_image_v2.npy')
    train_labels = np.load('./Defect_train_label_v2.npy')
    test_data = np.load('./Defect_test_image_v2.npy')
    test_labels = np.load('./Defect_test_label_v2.npy')

    train_filenames = []
    for idx, lab in enumerate(train_labels):
        imgName = 'waferDefect_train_' + DEFECT_CLASSES[lab[0]] + '_' + str(idx)+'.png'
        train_filenames.extend([imgName])

    test_filenames = []
    for idx, lab in enumerate(test_labels):
        imgName = 'waferDefect_test_' + DEFECT_CLASSES[lab[0]] + '_' + str(idx)+'.png'
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
    output_filename = f'{DATA_DIR}/{DATASET_NAME}.dataset'
    save_image_dataset(examples, output_filename)

    cnt = 0
    for class_idx in ['where']:
        fp_set_name = f'{DATA_DIR}/{DATASET_NAME}-{class_idx}.dataset'
        fp_openset_name = f'{DATA_DIR}/{DATASET_NAME}-not{class_idx}.dataset'
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

    # splits = [
    #     [3, 6, 7, 8],
    #     [1, 2, 4, 6],
    #     [2, 3, 4, 7],
    #     [0, 1, 2, 6],
    #     [4, 5, 6, 8],
    # ]

    splits = [
        [8, 9],
    ]

    for idx, split in enumerate(splits):
        # unknown_classes = [waferMap32_class(i) for i in split]
        unknown_classes = [DEFECT_CLASSES[i] for i in split]
        known_examples = [e for e in examples if e['label'] not in unknown_classes]
        unknown_examples = [e for e in examples if e['label'] in unknown_classes]
        known_filename = '{}/{}-split{}_known.dataset'.format(DATA_DIR, DATASET_NAME, idx)
        unknown_filename = '{}/{}-split{}_unknown.dataset'.format(DATA_DIR, DATASET_NAME, idx)
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
    # pixels = data*255
    # pixels = pixels.astype(np.uint8)
    # filename = filename
    pixels = data.reshape((512,512)).astype(np.uint8)
    Image.fromarray(pixels).save(filename)
    class_name = DEFECT_CLASSES[label[0]]
    return {
            'filename': os.path.join(DATASET_PATH, filename),
            'label': class_name,
    }

if __name__ == "__main__":
    main()