#! /bin/bash
# Break on any error
set -e

DATASET_DIR=/home/kcm/data/

# GAN_EPOCHS=1000
GAN_EPOCHS=200000
TRAIN_CLASSIFIER_EPOCH=1000
CHECK_EPOCH=8453
# CLASSIFIER_EPOCHS=3
# CF_COUNT=50

# Preprocess dataset
# if [ ! -f $DATASET_DIR/waferDefect_v4_2.dataset ]; then
#    python generativeopenset/datasets/download_defect_v4_2.py
# fi

# Calculate Mean and Standard Deviation of Data
# python generativeopenset/calculate_mean_std.py

# classification only basic classifier
python generativeopenset/train_basic_WResnet_classifier.py --epochs $TRAIN_CLASSIFIER_EPOCH

# Train the intial generative model (E+G+D) and the initial classifier (C_K)
# python generativeopenset/train_wgan.py --epochs $GAN_EPOCHS

# Train semi-supervised network(customed classifier) with trained generator(default=1000)
# python generativeopenset/train_VGG_classifier.py --epochs $TRAIN_CLASSIFIER_EPOCH

# Check the Classifier's Accuracy
# python generativeopenset/check_classifier_accuracy.py --epochs $CHECK_EPOCH
# python generativeopenset/check_classifier_confidence_map.py
# python generativeopenset/failed_data_check.py --epochs $CHECK_EPOCH
# python generativeopenset/check_with_excel.py --epochs $CHECK_EPOCH

# Check the Classfifier's Open Set classify performance
# python generativeopenset/evaluate_open_set_recognition.py --result_dir . --mode fuxin 
# python generativeopenset/evaluate_open_set_recognition.py --result_dir . --mode baseline
# python generativeopenset/evaluate_open_set_recognition.py --result_dir . --mode weibull 
# python generativeopenset/evaluate_open_set_recognition.py --result_dir . --mode SS_baseline 

# ./print_results.sh
