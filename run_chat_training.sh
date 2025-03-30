#!/bin/bash
# Example script to run chat fine-tuning with optimal parameters

# Choose a top open-source base model
MODEL_ID="meta-llama/Llama-3-8B" 

# Where to save the fine-tuned model
OUTPUT_DIR="./aegis-chat-model"

# Run training with Accelerate
python -m accelerate.commands.launch \
  --config_file accelerate_config.yaml \
  --multi_gpu \
  train_chat.py \
  --model_name_or_path $MODEL_ID \
  --dataset_name "HuggingFaceH4/ultrachat_200k" \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "steps" \
  --eval_steps 500 \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 3 \
  --learning_rate 1e-4 \
  --weight_decay 0.001 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --bf16 True \
  --report_to "tensorboard" \
  --use_peft True \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --load_in_4bit True

# Alternatively, you can use DeepSpeed
# deepspeed --num_gpus 8 train.py \
#   --deepspeed ds_config.json \
#   --model_name_or_path $MODEL_ID \
#   --dataset_name "HuggingFaceH4/ultrachat_200k" \
#   --output_dir $OUTPUT_DIR \
#   --use_peft True \
#   --lora_r 16 