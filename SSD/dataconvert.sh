#!/bin/bash 


DATASET_DIR=./VOC2007/
OUTPUT_DIR=./tfrecords

python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_2007_train \
    --output_dir=${OUTPUT_DIR}


# Dataset directory: ./VOC2007/
# Output directory: ./tfrecords
# >> Converting image 1/9963WARNING:tensorflow:From /home/zhangkailin/code/SSD-Tensorflow/datasets/pascalvoc_to_tfrecords.py:83: FastGFile.__init__ (from tensorflow.python.platform.gfile) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use tf.gfile.GFile.
# >> Converting image 9963/9963
# Finished converting the Pascal VOC dataset!