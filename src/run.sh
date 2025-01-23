#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export TRANSFORMERS_CACHE=./cache

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export DATASET_NAME="./PATH/DATASET/"
export OUTDIR="./PATH/OUTPUT/"

accelerate launch dcpo_trainer.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --caption_1="caption_1" \
  --caption_0="caption_0" \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=128 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=500 \
  --learning_rate=1e-08 \
  --scale_lr \
  --checkpointing_steps 500 \
  --output_dir=$OUTDIR \
  --mixed_precision="fp16" \
  --beta_dpo 500 \