#! /bin/bash
# Break on any error
set -e

DATASET_DIR=/home/kcm/data/

# Download any datasets not currently available
# TODO: do this in python, based on --dataset
# if [ ! -f $DATASET_DIR/svhn-split0a.dataset ]; then
#     python generativeopenset/datasets/download_svhn.py
# fi
if [ ! -f $DATASET_DIR/cifar10-split0a.dataset ]; then
   python generativeopenset/datasets/download_cifar10.py
fi
#if [ ! -f $DATASET_DIR/mnist-split0a.dataset ]; then
#    python generativeopenset/datasets/download_mnist.py
#fi
#if [ ! -f $DATASET_DIR/mnist32-split0a.dataset ]; then
#    python generativeopenset/datasets/download_mnist32.py
#fi
#if [ ! -f $DATASET_DIR/oxford102.dataset ]; then
#    python generativeopenset/datasets/download_oxford102.py
#fi
#if [ ! -f $DATASET_DIR/celeba.dataset ]; then
#    python generativeopenset/datasets/download_celeba. py
#fi
#if [ ! -f $DATASET_DIR/cifar100-animals.dataset ]; then
#    python generativeopenset/datasets/download_cifar100.py
#fi

#if [ ! -f $DATASET_DIR/wafer_map.dataset ]; then
#    python generativeopenset/datasets/download_wafer_map.py
#fi
# Hyperparameters
GAN_EPOCHS=30
CLASSIFIER_EPOCHS=3
CF_COUNT=50
#GENERATOR_MODE=open_set
GENERATOR_MODE=counterfactual

# Train the intial generative model (E+G+D) and the initial classifier (C_K)
python generativeopenset/train_gan.py --epochs $GAN_EPOCHS

# Baseline: Evaluate the standard classifier (C_k+1)
python generativeopenset/evaluate_classifier.py --result_dir . --mode baseline # 그저 쉽게 계산
python generativeopenset/evaluate_classifier.py --result_dir . --mode weibull # 

cp checkpoints/classifier_k_epoch_00${GAN_EPOCHS}.pth checkpoints/classifier_kplusone_epoch_00${GAN_EPOCHS}.pth

# Generate a number of counterfactual images (in the K+2 by K+2 square grid format)
python generativeopenset/generate_${GENERATOR_MODE}.py --result_dir . --count $CF_COUNT

# Automatically label the rightmost column in each grid (ignore the others)
python generativeopenset/auto_label.py --output_filename generated_images_${GENERATOR_MODE}.dataset

# Train a new classifier, now using the aux_dataset containing the counterfactuals
python generativeopenset/train_classifier.py --epochs $CLASSIFIER_EPOCHS --aux_dataset generated_images_${GENERATOR_MODE}.dataset

# Evaluate the C_K+1 classifier, trained with the augmented data
python generativeopenset/evaluate_classifier.py --result_dir . --mode fuxin

./print_results.sh
