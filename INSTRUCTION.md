# Step-by-Step Guide to Train, Save, and Deploy Your Model

## Part 1: Training the Model

### Prerequisites

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have GPU access (16GB+ VRAM recommended)

### Choose a Training Method

#### Option A: Standard Fine-tuning (QLoRA)

Best for general purpose fine-tuning:

1. Prepare your environment:

   ```bash
   mkdir -p logs
   ```

2. Run the training:

   ```bash
   bash train_advanced.sh
   ```

3. Monitor training progress:
   ```bash
   tensorboard --logdir ./aegis-advanced-model/runs
   ```

#### Option B: RLHF Training (DPO)

Best for aligning the model with human preferences:

1. Run RLHF training:

   ```bash
   bash train_rlhf.sh
   ```

2. Monitor training:
   ```bash
   tensorboard --logdir ./aegis-rlhf-model/runs
   ```

## Part 2: Preserving the Trained Model

### How Models Are Saved

Both training scripts automatically save:

- Model weights in the output directory
- LoRA adapters (small files containing the fine-tuned parameters)
- Training checkpoints at regular intervals
- Tokenizer files

### Saving Complete Models

1. After training completes, merge LoRA adapters with base model:

   ```bash
   python -c "
   from peft import PeftModel
   from transformers import AutoModelForCausalLM, AutoTokenizer

   # Load base model
   base_model = AutoModelForCausalLM.from_pretrained(
       'meta-llama/Llama-3-8B',
       torch_dtype='auto',
       device_map='auto'
   )
   tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3-8B')

   # Load adapter
   model_path = './aegis-advanced-model'  # or aegis-rlhf-model
   model = PeftModel.from_pretrained(base_model, model_path)

   # Merge adapter weights with base model
   merged_model = model.merge_and_unload()

   # Save complete model
   merged_model.save_pretrained('./aegis-merged-model')
   tokenizer.save_pretrained('./aegis-merged-model')
   "
   ```

2. Create model card with training details:
   ```bash
   echo "# AEGIS Fine-tuned Model
   - Base model: meta-llama/Llama-3-8B
   - Training method: QLoRA + DeepSpeed
   - Dataset: UltraChat 200k
   - Training date: $(date)
   " > ./aegis-merged-model/README.md
   ```

## Part 3: Deploying Your Model

### Local Deployment

1. Interactive chat interface (simplest):

   ```bash
   python interactive_chat.py --model_id ./aegis-merged-model --model_type llama3
   ```

2. Test with sample prompts:
   ```bash
   echo "Write a short story about AI" | python interactive_chat.py --model_id ./aegis-merged-model --no_interactive
   ```

### Server Deployment

1. Install server requirements:

   ```bash
   pip install gradio fastapi uvicorn
   ```

2. Create API server (save as `serve_model.py`):

   ```python
   from fastapi import FastAPI, Request
   from transformers import AutoModelForCausalLM, AutoTokenizer
   import torch
   import uvicorn
   import gradio as gr

   # Load your fine-tuned model
   model_path = "./aegis-merged-model"  # Path to your saved model
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       torch_dtype=torch.bfloat16,
       device_map="auto",
       load_in_4bit=True  # For memory efficiency
   )

   app = FastAPI()

   @app.post("/generate")
   async def generate(request: Request):
       data = await request.json()
       prompt = data.get("prompt", "")
       max_length = data.get("max_length", 512)

       inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

       with torch.no_grad():
           outputs = model.generate(
               inputs["input_ids"],
               max_new_tokens=max_length,
               temperature=0.7,
               top_p=0.95,
               do_sample=True
           )

       response = tokenizer.decode(outputs[0], skip_special_tokens=True)
       return {"response": response[len(prompt):]}

   # Add Gradio UI
   def predict(message, history):
       inputs = tokenizer(message, return_tensors="pt").to(model.device)
       with torch.no_grad():
           outputs = model.generate(
               inputs["input_ids"],
               max_new_tokens=512,
               temperature=0.7,
               do_sample=True
           )
       response = tokenizer.decode(outputs[0], skip_special_tokens=True)
       return response[len(message):]

   gr_interface = gr.ChatInterface(
       predict,
       title="AEGIS LLM",
       description="Interact with your fine-tuned model"
   )

   app = gr.mount_gradio_app(app, gr_interface, path="/")

   if __name__ == "__main__":
       uvicorn.run(app, host="0.0.0.0", port=8000)
   ```

3. Run the server:

   ```bash
   python serve_model.py
   ```

4. Access:
   - Web UI: http://localhost:8000
   - API endpoint: http://localhost:8000/generate (POST)

### Cloud Deployment

1. Create Dockerfile:

   ```bash
   echo 'FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY ./aegis-merged-model /app/model
   COPY serve_model.py .

   EXPOSE 8000

   CMD ["python", "serve_model.py"]' > Dockerfile
   ```

2. Build and push Docker image:

   ```bash
   docker build -t aegis-llm:latest .
   docker tag aegis-llm:latest your-registry/aegis-llm:latest
   docker push your-registry/aegis-llm:latest
   ```

3. Deploy to cloud:
   - AWS: Use SageMaker or EC2 with GPU
   - Azure: Use Azure ML or AKS with GPU
   - GCP: Use Vertex AI or GKE with GPU

## Part 4: Optimizations for Production

1. Quantize for faster inference:

   ```bash
   # Install optimum
   pip install optimum auto-gptq

   # Quantize model to 4-bit
   python -c "
   from optimum.gptq import GPTQQuantizer
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model_id = './aegis-merged-model'
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   model = AutoModelForCausalLM.from_pretrained(model_id)

   quantizer = GPTQQuantizer(bits=4, dataset='c4', tokenizer=tokenizer)
   quantized_model = quantizer.quantize_model(model)

   # Save quantized model
   quantized_model.save_pretrained('./aegis-quantized')
   tokenizer.save_pretrained('./aegis-quantized')
   "
   ```

2. Use vLLM for faster inference:

   ```bash
   pip install vllm

   # Run vLLM server
   python -m vllm.entrypoints.api_server --model ./aegis-merged-model --host 0.0.0.0 --port 8000
   ```
