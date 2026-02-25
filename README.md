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


