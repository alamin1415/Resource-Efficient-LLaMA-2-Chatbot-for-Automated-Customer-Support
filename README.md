markdown
# 🦙 Llama-2-7B Fine-Tuning with QLoRA

This repository contains a complete implementation of fine-tuning Llama-2-7B using QLoRA (Quantized Low-Rank Adaptation) on a custom question-answering dataset. The entire training was performed on Google Colab with a T4 GPU (15GB VRAM), demonstrating efficient parameter-efficient fine-tuning for large language models.

## 📋 Project Overview

- **Model:** Llama-2-7B (TinyPixel/Llama-2-7B-bf16-sharded)
- **Method:** QLoRA (4-bit quantization + LoRA)
- **Hardware:** Google Colab T4 GPU (15GB VRAM)
- **Training Steps:** 100 steps
- **Final Loss:** 0.0685
- **Dataset:** Custom JSON with question-answer pairs (13 examples)

## ✨ Key Features

- **Memory Efficient Training:** Reduced model memory from ~28GB to under 8GB using:
  - 4-bit quantization (NF4 format)
  - LoRA adapters (r=64, alpha=16)
  - Gradient checkpointing
  
- **Complete Pipeline:**
  - Data loading from local JSON
  - Model quantization with BitsAndBytes
  - PEFT configuration with LoRA
  - Training with SFTTrainer
  - Model saving and deployment to Hugging Face Hub

- **Training Optimizations:**
  - Mixed precision training (fp16)
  - Gradient accumulation
  - Paged AdamW optimizer
  - Layer norm upcasting for stability

## 🛠️ Technologies Used

- **Python 3.12**
- **PyTorch 2.x**
- **Hugging Face Ecosystem:**
  - Transformers
  - PEFT (Parameter-Efficient Fine-Tuning)
  - TRL (Transformer Reinforcement Learning)
  - Datasets
  - BitsAndBytes (4-bit quantization)
- **Weights & Biases** (experiment tracking)
- **Google Colab**

## 📊 Training Results

The model converged rapidly with the following loss progression:

| Step | Training Loss |
|------|---------------|
| 10   | 1.8699 |
| 20   | 0.8748 |
| 30   | 0.2811 |
| 40   | 0.1220 |
| 50   | 0.0801 |
| 60   | 0.0713 |
| 70   | 0.0694 |
| 80   | 0.0688 |
| 90   | 0.0686 |
| 100  | 0.0685 |

**Final metrics:**
- Training runtime: 373.79 seconds
- Samples per second: 4.28
- Steps per second: 0.268
- Total loss: 0.3574

## 📁 Dataset Structure

The dataset is a JSON file containing question-answer pairs:

```json
{
  "questions": [
    {
      "question": "How can I create an account?",
      "answer": "To create an account, click on the 'Sign Up' button..."
    },
    {
      "question": "What payment methods do you accept?",
      "answer": "We accept major credit cards, debit cards, and PayPal..."
    }
    // ... more examples
  ]
}
🚀 How to Use
Installation
bash
pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
pip install -q datasets bitsandbytes einops wandb
Load the Fine-Tuned Model
python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Load base model with quantization
model_name = "TinyPixel/Llama-2-7B-bf16-sharded"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the fine-tuned adapter
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True
    ),
    "outputs"  # or your adapter path
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
Inference Example
python
def generate_answer(question):
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=2.0,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
question = "How can I create an account?"
answer = generate_answer(question)
print(answer)
📦 Model Deployment
The fine-tuned model is available on Hugging Face Hub:

Repository: alamin1415/llama2-qlora-finetunined-french

To load directly from Hugging Face:

python
from peft import PeftModel

model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config),
    "alamin1415/llama2-qlora-finetunined-french"
)
🔧 Configuration Details
QLoRA Configuration
LoRA rank (r): 64

LoRA alpha: 16

LoRA dropout: 0.1

Bias: None

Task type: CAUSAL_LM

Training Arguments
Batch size: 4 (per device)

Gradient accumulation: 4 steps

Learning rate: 2e-4

Optimizer: paged_adamw_32bit

Max steps: 100

Warmup ratio: 0.03

LR scheduler: constant

Max gradient norm: 0.3
