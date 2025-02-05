#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export TRANSFORMERS_CACHE=./cache

export DATASET_NAME="./PATH/DATASET/"
export OUTDIR="./PATH/OUTPUT/"

python perturb_caption.py \
  --hf_file=$DATASET_NAME \
  --out_folder=$OUTDIR \ 