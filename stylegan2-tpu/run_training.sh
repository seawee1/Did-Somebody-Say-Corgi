#!/bin/sh

export NOISY=0
export DEBUG=0
export LABEL_SIZE=0
export BUCKET=gs://stylegan2-bucket
export MODEL_DIR=${BUCKET}/model
export DATA_DIR=${BUCKET}/data
export DATASET=256
export BATCH_PER=4
export BATCH_SIZE=32
export RESOLUTION=256
export TPU_NAME=stylegan2

python3.6 run_training.py --result-dir=${MODEL_DIR} --data-dir=${DATA_DIR} --dataset=256 --config=config-f --num-gpus=8 --mirror-augment=true
