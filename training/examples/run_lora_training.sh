#!/bin/bash
# Example script for LoRA fine-tuning on a custom dataset

# Base model
MODEL_ID="meta-llama/Llama-3-8B"

# Output directory
OUTPUT_DIR="./outputs/aegis-lora-model"

# Run training with PEFT/LoRA
python ../../train.py \
  --model_name_or_path $MODEL_ID \
  --train_file ../../training/datasets/custom_data.jsonl \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --save_strategy "steps" \
  --save_steps 200 \
  --bf16 True \
  --use_peft True \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --load_in_4bit True 