#!/usr/bin/env python
"""
Aegis Training Script - Fine-tune language models with state-of-the-art techniques

This script provides a flexible training pipeline using the best open-source tools:
- Hugging Face Transformers, Accelerate, and PEFT for efficient training
- DeepSpeed and FSDP for distributed training
- Support for quantization, mixed precision, and gradient accumulation
"""

import os
import argparse
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": "Use Parameter-Efficient Fine-Tuning techniques"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "Lora r dimension"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "Lora alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Lora dropout"}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8-bit precision"}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load model in 4-bit precision"}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate on (a text file)."},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    text_column: Optional[str] = field(
        default="text",
        metadata={
            "help": "The name of the column in the datasets containing the text data."},
    )


def main():
    """Main training function."""
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model
    set_seed(training_args.seed)

    # Load dataset
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        raw_datasets = load_dataset("text", data_files=data_files)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Ensure tokenizer has padding token and eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare model
    device_map = "auto"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Determine quantization parameters
    quantization_config = None
    if model_args.load_in_8bit:
        quantization_config = {"load_in_8bit": True}
    elif model_args.load_in_4bit:
        quantization_config = {"load_in_4bit": True,
                               "bnb_4bit_compute_dtype": torch_dtype}

    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    # Apply PEFT if requested
    if model_args.use_peft:
        logger.info("Applying LoRA adapters...")
        # Prepare the model for kbit training if using quantization
        if model_args.load_in_8bit or model_args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                            "W_pack", "wk", "wv", "wq", "wo"],
            bias="none",
            fan_in_fan_out=False,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples[data_args.text_column],
            max_length=data_args.max_seq_length,
            truncation=True
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    # Configure data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation", None),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate the model
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
