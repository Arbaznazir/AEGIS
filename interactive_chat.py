import os
import argparse
from model_utils import load_model, advanced_generate, chat_format
from config import GENERATION_PRESETS, DEFAULT_SYSTEM_PROMPT


def main():
    """
    Main function for interactive chat with the model.
    """
    parser = argparse.ArgumentParser(
        description="Interactive chat with language model")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3-8B-Instruct",
                        help="Hugging Face model ID")
    parser.add_argument("--cpu_only", action="store_true",
                        help="Force CPU usage")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Penalty for repeated tokens")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Number of beams for beam search (1 = no beam search)")
    parser.add_argument("--preset", type=str, choices=list(GENERATION_PRESETS.keys()),
                        help="Use a predefined generation preset (overrides other parameters)")
    parser.add_argument("--model_type", type=str, default="llama3",
                        help="Model type for chat formatting (llama3, mistral, llama)")
    args = parser.parse_args()

    # Set environment variable to use BF16 precision if available
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    # Load model and tokenizer
    model, tokenizer = load_model(
        args.model_id, use_gpu=not args.cpu_only, quantize="4bit")

    # System prompt
    system_prompt = DEFAULT_SYSTEM_PROMPT

    # Initialize chat history
    messages = []

    print("\n" + "="*60)
    print("Aegis Model - Advanced NLP System")
    print("Powered by Hugging Face Transformers - The Leading NLP Framework")
    print("="*60)
    print(f"\nModel: {args.model_id}")
    print("Type '/exit' to quit, '/clear' to start a new conversation, '/params' to show current parameters")
    print("Type '/preset <n>' to use a generation preset (creative, precise, or balanced)")
    print("Type '/model_type <type>' to change the chat formatting (llama3, mistral, llama)\n")

    # Store generation parameters for easy access
    gen_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams
    }

    # Apply preset if specified
    if args.preset and args.preset in GENERATION_PRESETS:
        gen_params.update(GENERATION_PRESETS[args.preset])
        print(f"Using '{args.preset}' generation preset")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "/exit":
            print("Goodbye!")
            break

        if user_input.lower() == "/clear":
            messages = []
            print("Conversation cleared.")
            continue

        if user_input.lower() == "/params":
            print("\nCurrent generation parameters:")
            for param, value in gen_params.items():
                print(f"{param}: {value}")
            continue

        if user_input.lower().startswith("/temp "):
            try:
                temp = float(user_input.split(" ")[1])
                if 0 <= temp <= 2:
                    gen_params["temperature"] = temp
                    print(f"Temperature set to {temp}")
                else:
                    print("Temperature must be between 0 and 2")
            except:
                print("Invalid temperature value")
            continue

        if user_input.lower().startswith("/preset "):
            preset_name = user_input.lower().split(" ")[1]
            if preset_name in GENERATION_PRESETS:
                gen_params.update(GENERATION_PRESETS[preset_name])
                print(f"Using '{preset_name}' generation preset")
            else:
                print(
                    f"Unknown preset '{preset_name}'. Available presets: {', '.join(GENERATION_PRESETS.keys())}")
            continue

        if user_input.lower().startswith("/model_type "):
            model_type = user_input.lower().split(" ")[1]
            if model_type in ["llama3", "mistral", "llama"]:
                # Update the model type
                print(f"Changed chat format to {model_type}")
                continue
            else:
                print(
                    f"Unknown model type '{model_type}'. Available types: llama3, mistral, llama")
                continue

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Format the conversation
        prompt = chat_format(messages, system_prompt,
                             model_type=args.model_type)

        # Generate response using advanced generation
        response = advanced_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=gen_params["max_new_tokens"],
            temperature=gen_params["temperature"],
            top_p=gen_params["top_p"],
            top_k=gen_params["top_k"],
            repetition_penalty=gen_params["repetition_penalty"],
            num_beams=gen_params["num_beams"]
        )

        # Add assistant response to history
        messages.append({"role": "assistant", "content": response})

        # Print the response
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()
