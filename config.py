"""
Configuration settings for the model.
"""

# Default model settings
# Small model that works well on Colab
DEFAULT_MODEL_ID = "meta-llama/Llama-3-8B-Instruct"
BACKUP_MODEL_IDS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2-7B-Instruct"
]

# High-quality model options (if resources available)
PREMIUM_MODEL_IDS = [
    "meta-llama/Llama-3-70B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Qwen/Qwen2-72B-Instruct",
    "THUDM/glm-4-9b-chat"
]

# Generation settings
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
DO_SAMPLE = True
TOP_P = 0.95
TOP_K = 50
REPETITION_PENALTY = 1.1
MIN_LENGTH = 0
NO_REPEAT_NGRAM_SIZE = 3
NUM_BEAMS = 1
NUM_RETURN_SEQUENCES = 1

# Generation preset configurations
GENERATION_PRESETS = {
    "creative": {
        "temperature": 0.9,
        "top_p": 0.92,
        "top_k": 60,
        "repetition_penalty": 1.05,
        "num_beams": 1,
    },
    "precise": {
        "temperature": 0.3,
        "top_p": 0.85,
        "top_k": 40,
        "repetition_penalty": 1.2,
        "num_beams": 2,
    },
    "balanced": {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "num_beams": 1,
    }
}

# System prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. 
Always answer as helpfully as possible, while being safe.
If you don't know the answer to a question, please don't share false information."""

# Memory settings for different devices
MEMORY_CONFIG = {
    "low": {
        "device_map": "auto",
        "max_memory": {0: "4GiB"}
    },
    "medium": {
        "device_map": "auto",
        "max_memory": {0: "8GiB"}
    },
    "high": {
        "device_map": "auto",
        "max_memory": {0: "16GiB"}
    }
}
