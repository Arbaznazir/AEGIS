import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_utils import advanced_generate


def main():
    """
    Simple test of the advanced generation algorithm.
    """
    print("Loading a small model for testing...")

    # Use a small modern model for testing
    model_id = "microsoft/Phi-3-mini-4k-instruct"  # Small but modern model

    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # Test prompts
    test_prompts = [
        "The best thing about artificial intelligence is",
        "In five years, the technology industry will",
        "The most important skill for the future is"
    ]

    # Generation settings for comparison
    settings = [
        {
            "name": "Default",
            "params": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.0,
                "max_new_tokens": 50
            }
        },
        {
            "name": "Creative",
            "params": {
                "temperature": 0.9,
                "top_p": 0.92,
                "top_k": 60,
                "repetition_penalty": 1.05,
                "max_new_tokens": 50
            }
        },
        {
            "name": "Precise",
            "params": {
                "temperature": 0.3,
                "top_p": 0.85,
                "top_k": 40,
                "repetition_penalty": 1.2,
                "max_new_tokens": 50
            }
        }
    ]

    # Run tests
    for prompt in test_prompts:
        print(f"\n\nTesting prompt: '{prompt}'")

        for setting in settings:
            print(f"\n{setting['name']} setting:")
            response = advanced_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                **setting["params"]
            )
            print(f"Response: {response}")

    print("\nTest completed!")


if __name__ == "__main__":
    main()
