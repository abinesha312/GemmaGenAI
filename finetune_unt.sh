#!/usr/bin/env bash
# This script runs Gemma 3 27B fine-tuning with UNT-specific data
set -euo pipefail

# Required packages (install these manually before running):
# pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# pip install transformers==4.36.2
# pip install peft==0.7.1
# pip install accelerate==0.26.1
# pip install bitsandbytes==0.41.2.post2
# pip install datasets==2.16.0
# pip install sentencepiece==0.1.99
# pip install protobuf==4.25.1
# pip install tqdm==4.66.1
# pip install safetensors==0.4.1

# Set environment variables for fine-tuning
export DATA_PATH="/home/models/FAISS_INGEST/scraped_data.json"
export OUTPUT_DIR="./unt-gemma-3-finetuned"
export EPOCHS=5  # More epochs for better learning
export BATCH_SIZE=1
export GRAD_ACCUM=4
export LORA_RANK=16
export LORA_ALPHA=32
export MAX_SEQ_LENGTH=4096
export SAVE_STEPS=50  # Save more frequently
export LR=2e-5

# Run the fine-tuning script
./run_finetune.sh

# Optional: Test the model after training
echo ""
echo "Testing the fine-tuned model with a UNT-specific question..."
python3 -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model and adapter
model = AutoModelForCausalLM.from_pretrained('google/gemma-3-27b-it', device_map='auto', trust_remote_code=True)
model = PeftModel.from_pretrained(model, '$OUTPUT_DIR')
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-27b-it', trust_remote_code=True)

# Prepare a UNT-specific question
prompt = '<|user|>\nWhat dining options are available at UNT?\n<|assistant|>'

# Generate a response
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=256,
    temperature=0.2,
    do_sample=True,
)

# Print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)
" 