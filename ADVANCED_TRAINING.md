# Advanced Training Guide for AEGIS

This guide explains the state-of-the-art training methods implemented in this project for fine-tuning open-source LLMs.

## Training Methods

We've implemented multiple cutting-edge training approaches:

### 1. QLoRA with DeepSpeed ZeRO-3 (Highest Quality)

This approach combines quantized LoRA (QLoRA) with DeepSpeed ZeRO-3 optimization for maximum efficiency. This method allows fine-tuning models like Llama 3 70B on consumer hardware:

```bash
bash train_advanced.sh
```

**Key features:**

- 4-bit quantization to reduce memory requirements by 8x
- DeepSpeed ZeRO-3 for efficient memory management
- LoRA rank 32 with alpha 64 for high-quality adaptation
- Flash Attention 2 for faster training with longer context
- Cosine learning rate schedule with restarts

### 2. RLHF with Direct Preference Optimization (Most Advanced)

This implements Reinforcement Learning from Human Feedback using the Direct Preference Optimization (DPO) algorithm, which is the state-of-the-art approach for aligning language models with human preferences:

```bash
bash train_rlhf.sh
```

**Key features:**

- DPO algorithm - more efficient than traditional RLHF (PPO)
- No reward model training needed
- Trained directly on human preference data
- Uses the Anthropic Helpful-Harmless dataset
- Better alignment with human values and instructions

## Hardware Requirements

For optimal training, we recommend:

- **GPU**: NVIDIA RTX 4090, A100, or A6000 (24GB+ VRAM)
- **Memory**: 32GB+ RAM
- **Storage**: 100GB+ SSD space

For larger models (70B+), we recommend:

- Multiple GPUs (2-8 depending on model size)
- 64GB+ RAM
- 500GB+ SSD space

## Training Datasets

The scripts support various datasets:

1. **UltraChat 200k**: High-quality conversational data (default for QLoRA training)
2. **Anthropic Helpful-Harmless**: Specialized RLHF data (default for RLHF training)
3. **Custom datasets**: You can use your own data by replacing the dataset parameters

## Customization Options

You can customize the training by modifying the following parameters:

- **Model**: Change `MODEL_ID` to any Hugging Face model
- **Learning rate**: Adjust based on model size (smaller for larger models)
- **Batch size**: Reduce for larger models or increase for smaller ones
- **LoRA rank**: Higher values (32-64) for better quality, lower (8-16) for efficiency

## Results and Evaluation

After training, models will be saved to the specified output directory. You can evaluate them using:

```bash
python interactive_chat.py --model_id ./aegis-advanced-model
```

Or with the RLHF model:

```bash
python interactive_chat.py --model_id ./aegis-rlhf-model
```

## Troubleshooting

- **Out of memory errors**: Reduce batch size, use gradient checkpointing, or try 4-bit quantization
- **Training crashes**: Check for compatible versions in requirements.txt
- **Poor results**: Increase LoRA rank/alpha, use longer training, or try a different base model

## References

Our implementation builds on these cutting-edge research papers:

1. QLoRA: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
2. DPO: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
3. Flash Attention 2: [Flash Attention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
