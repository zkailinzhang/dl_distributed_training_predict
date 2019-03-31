#!/bin/bash

DATASET_DIR=./tfrecords
TRAIN_DIR=./logs/
CHECKPOINT_PATH=./checkpoints/ssd_300_vgg.ckpt
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --batch_size=8




    # Cannot assign a device for operation global_step: Operation was explicitly assigned to /device:GPU:0 but available devices are 
    # [ /job:localhost/replica:0/task:0/device:CPU:0, /job:localhost/replica:0/task:0/device:XLA_CPU:0 ]. 
    # Make sure the device specification refers to a valid device.


    # 2019-03-29 20:50:01.448373: E tensorflow/core/common_runtime/executor.cc:623] Executor failed to create kernel. 
    # Invalid argument: Default MaxPoolingOp only supports NHWC on device type CPU

    #  Allocation of 737280000 exceeds 10% of system memory.
    #  batch =32