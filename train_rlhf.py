#!/usr/bin/env python
"""
Advanced RLHF Training with TRL - Transformer Reinforcement Learning
Implements state-of-the-art DPO (Direct Preference Optimization) training for top LLMs
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

# Define command-line arguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Load model in 4-bit precision"}
    )
    use_rlhf: bool = field(
        default=True,
        metadata={"help": "Use RLHF training with DPO"}
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Use Flash Attention for faster training"}
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        default="Anthropic/hh-rlhf",
        metadata={"help": "The name of the dataset to use for RLHF training"}
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "The split of the dataset to use"}
    )
    max_seq_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length to use"}
    )


def main():
    # Parse arguments
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Configure hardware optimizations
    if model_args.use_flash_attn:
        try:
            from flash_attn import flash_attn_func
            os.environ["TRANSFORMERS_USE_FLASH_ATTENTION_2"] = "true"
            print("Flash Attention 2 enabled")
        except ImportError:
            print("Flash Attention not available, falling back to standard attention")

    # Calculate quantization parameters
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    quant_config = None

    if model_args.use_4bit:
        quant_config = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": compute_dtype,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        }

    # Load model and tokenizer
    print(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        quantization_config=quant_config if model_args.use_4bit else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA for efficient fine-tuning
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "W_pack", "wk", "wv", "wq", "wo"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load RLHF dataset
    print(f"Loading dataset: {data_args.dataset_name}")

    # Configure DPO training (state-of-the-art RLHF implementation)
    dpo_config = DPOConfig(
        max_length=data_args.max_seq_length,
        max_prompt_length=data_args.max_seq_length // 2,
        beta=0.1,  # DPO temperature parameter
        peft_config=peft_config,
    )

    # Initialize the DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=data_args.dataset_name,
        beta=0.1,
        max_length=data_args.max_seq_length,
        max_prompt_length=data_args.max_seq_length // 2,
        peft_config=peft_config,
    )

    # Start training
    print("Starting DPO training...")
    trainer.train()

    # Save the model
    trainer.save_model(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
