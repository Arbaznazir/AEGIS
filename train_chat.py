#!/usr/bin/env python
"""
Aegis Chat Tuning Script - Fine-tune LLMs for conversational use

This script provides a specialized training pipeline for chat/instruction tuning:
- Support for chat templates and instruction formats
- Optimized for conversational datasets
- Includes best practices for instruction-tuning
"""

import os
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
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_peft: bool = field(
        default=True,
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
        default=True,
        metadata={"help": "Load model in 4-bit precision"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
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
    max_seq_length: int = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    instruction_column: str = field(
        default="instruction",
        metadata={
            "help": "The name of the column in the datasets containing the instructions."},
    )
    input_column: str = field(
        default="input",
        metadata={
            "help": "The name of the column in the datasets containing the inputs."},
    )
    output_column: str = field(
        default="output",
        metadata={
            "help": "The name of the column in the datasets containing the outputs."},
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={
            "help": "The chat template to use (will use model's default if not specified)"},
    )


def main():
    """Main training function for chat/instruction tuning."""
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

        extension = data_args.train_file.split(
            ".")[-1] if data_args.train_file else "json"
        raw_datasets = load_dataset(extension, data_files=data_files)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Set chat template if provided
    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template

    # Ensure tokenizer has padding token
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
        quantization_config = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch_dtype,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        }

    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    # Apply PEFT/LoRA if requested
    if model_args.use_peft:
        logger.info("Applying LoRA adapters...")
        # Prepare the model for kbit training if using quantization
        if model_args.load_in_8bit or model_args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        # Configure LoRA - adapt target modules to model architecture
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj",
                            "o_proj", "gate_proj", "up_proj", "down_proj",
                            "W_pack", "o_proj", "wk", "wv", "wq", "wo"],
            bias="none",
            fan_in_fan_out=False,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Tokenize datasets using conversation templates
    def prepare_training_data(examples):
        messages = []
        for i in range(len(examples[data_args.instruction_column])):
            instruction = examples[data_args.instruction_column][i]
            input_text = examples.get(data_args.input_column, [
                                      ""] * len(examples[data_args.instruction_column]))[i]
            output = examples[data_args.output_column][i]

            # Create chat message format
            if input_text:
                messages.append([
                    {"role": "user", "content": f"{instruction}\n{input_text}"},
                    {"role": "assistant", "content": output}
                ])
            else:
                messages.append([
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output}
                ])

        # Apply chat template to create training examples
        tokenized_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            padding="max_length",
            max_length=data_args.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_inputs["input_ids"].clone(),
        }

    # Process datasets
    processed_datasets = raw_datasets.map(
        prepare_training_data,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing and formatting conversations",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets.get("validation", None),
        tokenizer=tokenizer,
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
