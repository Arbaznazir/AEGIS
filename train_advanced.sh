#!/bin/bash
# Advanced training script using QLoRA + DeepSpeed + Flash Attention 2

# Top open-source model
MODEL_ID="meta-llama/Llama-3-8B"
OUTPUT_DIR="./aegis-advanced-model"

# Run the training with state-of-the-art settings
deepspeed train_chat.py \
  --deepspeed training_configs/ds_config_zero3.json \
  --model_name_or_path $MODEL_ID \
  --dataset_name "HuggingFaceH4/ultrachat_200k" \
  --dataset_config_name default \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --evaluation_strategy "steps" \
  --eval_steps 500 \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 3 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type "cosine_with_restarts" \
  --logging_steps 10 \
  --report_to "tensorboard" \
  --use_peft True \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --load_in_4bit True \
  --max_seq_length 4096 \
  --preprocessing_num_workers 16 \
  --bf16 True
