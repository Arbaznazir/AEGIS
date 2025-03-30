"""
Aegis Model - Advanced NLP System
Powered by Hugging Face Transformers - The Leading Open-Source NLP Framework

This module implements enhanced text generation algorithms using the world's most
powerful open-source NLP library: Hugging Face Transformers.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline


def load_model(model_id, use_gpu=True, max_memory=None, quantize=None):
    """
    Load a language model from Hugging Face.

    Args:
        model_id (str): The Hugging Face model ID
        use_gpu (bool): Whether to use GPU if available
        max_memory (dict, optional): Memory constraints for specific devices
        quantize (str, optional): Quantization type ('4bit', '8bit', or None)

    Returns:
        tuple: (model, tokenizer)
    """
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    print(f"Loading model {model_id} on {device}...")

    kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
    }

    # Configure quantization if specified
    if quantize == "4bit":
        kwargs["quantization_config"] = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        }
    elif quantize == "8bit":
        kwargs["quantization_config"] = {"load_in_8bit": True}

    if max_memory:
        kwargs["max_memory"] = max_memory

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **kwargs
    )

    # Ensure tokenizer has pad token for proper batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_pipeline(model, tokenizer):
    """
    Create a text generation pipeline.

    Args:
        model: The language model
        tokenizer: The tokenizer

    Returns:
        TextGenerationPipeline: A generation pipeline
    """
    return TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer
    )


def generate_response(pipeline, prompt, max_new_tokens=512, temperature=0.7, do_sample=True,
                      top_p=0.95, top_k=50, repetition_penalty=1.1, min_length=0,
                      no_repeat_ngram_size=0):
    """
    Generate a response to a prompt using improved parameters.

    Args:
        pipeline: The text generation pipeline
        prompt (str): The input prompt
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature
        do_sample (bool): Whether to use sampling
        top_p (float): Nucleus sampling parameter
        top_k (int): Top-k sampling parameter
        repetition_penalty (float): Penalty for token repetition
        min_length (int): Minimum length of the generated text
        no_repeat_ngram_size (int): Size of n-grams to prevent repetition

    Returns:
        str: The generated response
    """
    result = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        min_length=min_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=pipeline.tokenizer.eos_token_id,
    )

    return result[0]["generated_text"][len(prompt):]


def advanced_generate(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7,
                      top_p=0.95, top_k=50, repetition_penalty=1.1,
                      min_length=0, num_beams=1, num_return_sequences=1):
    """
    Advanced generation function that gives more control and better performance.
    Uses batch processing and model.generate() directly for better efficiency.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt (str): The input prompt
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature
        top_p (float): Nucleus sampling parameter
        top_k (int): Top-k sampling parameter  
        repetition_penalty (float): Penalty for token repetition
        min_length (int): Minimum length of the generated text
        num_beams (int): Number of beams for beam search
        num_return_sequences (int): Number of sequences to return

    Returns:
        str: The generated response
    """
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Set up generation parameters
    with torch.no_grad():
        # Use efficient generation with proper parameters
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_length=min_length+input_ids.shape[1],
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and return the response, removing the input prompt
    if num_return_sequences == 1:
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
    else:
        decoded = [tokenizer.decode(
            output, skip_special_tokens=True) for output in outputs]
        prompt_len = len(tokenizer.decode(
            input_ids[0], skip_special_tokens=True))
        return [text[prompt_len:] for text in decoded]


def chat_format(messages, system_prompt=None, model_type="llama"):
    """
    Format a chat conversation for the model based on its type.

    Args:
        messages (list): List of message dictionaries with "role" and "content"
        system_prompt (str, optional): Optional system prompt
        model_type (str): Model type for formatting (llama, mistral, openchat)

    Returns:
        str: Formatted chat string
    """
    if model_type == "llama3":
        # Llama 3 format
        formatted = ""
        if system_prompt:
            formatted += f"<|system|>\n{system_prompt}\n"

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                formatted += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}\n"
            elif role == "system":
                formatted += f"<|system|>\n{content}\n"

        # Add final assistant marker for generation
        formatted += "<|assistant|>\n"
        return formatted

    elif model_type == "mistral":
        # Mistral format
        formatted = ""
        if system_prompt:
            formatted += f"<s>[INST] {system_prompt} [/INST]</s>\n"

        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                formatted += f"<s>[INST] {content} [/INST]"
            elif role == "assistant":
                formatted += f" {content}</s>\n"

        return formatted

    else:
        # Default format (original Llama 2 style)
        formatted = ""
        if system_prompt:
            formatted += f"<|system|>\n{system_prompt}\n\n"

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                formatted += f"<|user|>\n{content}\n\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}\n\n"
            elif role == "system":
                formatted += f"<|system|>\n{content}\n\n"

        # Add final assistant marker for generation
        formatted += "<|assistant|>\n"
        return formatted
