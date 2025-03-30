#!/bin/bash
# Advanced RLHF training script using DPO (Direct Preference Optimization)
# This implements state-of-the-art training algorithm for LLMs

# Set model
MODEL_ID="meta-llama/Llama-3-8B"
OUTPUT_DIR="./aegis-rlhf-model"

# Run RLHF training using DPO algorithm
python train_rlhf.py \
  --model_name_or_path $MODEL_ID \
  --dataset_name "Anthropic/hh-rlhf" \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-6 \
  --lr_scheduler_type "cosine" \
  --warmup_steps 100 \
  --weight_decay 0.05 \
  --optim "adamw_torch" \
  --logging_steps 10 \
  --save_strategy "steps" \
  --save_steps 500 \
  --evaluation_strategy "steps" \
  --eval_steps 500 \
  --bf16 \
  --max_seq_length 4096 \
  --use_4bit True \
  --use_flash_attn True \
  --gradient_checkpointing True \
  --report_to "tensorboard" 