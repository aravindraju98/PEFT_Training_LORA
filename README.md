# PEFT LoRA Fine-tuning for Customer Support

This repository contains a notebook demonstrating Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation) for training a language model on customer support data.

## Overview

This project showcases how to efficiently fine-tune large language models using LoRA, which dramatically reduces the number of trainable parameters while maintaining performance. The example demonstrates fine-tuning a LLaMA model for customer support responses.

## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that:
- Reduces trainable parameters by ~90-99%
- Maintains model performance
- Enables faster training with lower memory requirements
- Allows multiple task-specific adapters to be stored separately

## Features

- **Model**: LLaMA-7B base model (`decapoda-research/llama-7b-hf`)
- **Task**: Customer support response generation
- **Technique**: LoRA adaptation with PEFT library
- **Dataset**: Custom customer support conversations
- **Training**: Efficient fine-tuning with minimal resources

## Requirements

```bash
pip install transformers datasets peft accelerate
```

## Configuration

### LoRA Parameters
- **Rank (r)**: 8 - Controls the dimensionality of the low-rank matrices
- **Alpha**: 16 - Scaling parameter for LoRA weights
- **Dropout**: 0.1 - Regularization to prevent overfitting
- **Target Modules**: `["q_proj", "v_proj"]` - Attention layers to adapt

### Training Parameters
- **Batch Size**: 2 per device
- **Epochs**: 3
- **Learning Rate**: 2e-4
- **Max Length**: 128 tokens
- **Mixed Precision**: FP16 enabled

## Dataset Format

The training data follows this structure:
```python
{
    "instruction": "Customer: My package arrived damaged. Help me.",
    "response": "I'm sorry to hear that. We'll arrange a replacement immediately."
}
```

## Usage

1. **Install Dependencies**:
   ```bash
   pip install transformers datasets peft accelerate
   ```

2. **Run the Notebook**:
   Open `PEFT_QLORA_training(1).ipynb` and execute all cells

3. **Customize the Data**:
   Replace the example `data` list with your own customer support conversations

4. **Adjust Parameters**:
   Modify LoRA configuration and training arguments as needed

## File Structure

```
New folder/
├── PEFT_QLORA_training(1).ipynb    # Main training notebook
├── README.md                        # This file
└── customer_support_lora/           # Output directory (created after training)
    ├── adapter_config.json
    ├── adapter_model.bin
    └── tokenizer files
```

## Key Benefits of This Approach

1. **Memory Efficient**: Only fine-tune a small subset of parameters
2. **Fast Training**: Reduced computational requirements
3. **Modular**: LoRA adapters can be swapped without affecting base model
4. **Cost-Effective**: Lower GPU memory and training time requirements

## Customization Options

### Different Models
Replace `model_name` with other compatible models:
- `microsoft/DialoGPT-medium`
- `facebook/blenderbot-400M-distill`
- `gpt2`

### Different Tasks
Adjust the `task_type` in LoRA config:
- `TaskType.SEQ_2_SEQ_LM` for sequence-to-sequence
- `TaskType.QUESTION_ANS` for question answering
- `TaskType.TOKEN_CLS` for token classification

### Target Modules
Common target modules for different architectures:
- **LLaMA**: `["q_proj", "v_proj", "k_proj", "o_proj"]`
- **GPT-2**: `["c_attn", "c_proj"]`
- **T5**: `["q", "v", "k", "o"]`

## Results

After training, the model will be saved to `./customer_support_lora/` and can be loaded using:

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, "./customer_support_lora")
```

## Next Steps

- Expand the dataset with more diverse customer support scenarios
- Experiment with different LoRA ranks and target modules
- Implement evaluation metrics for response quality
- Try QLoRA for even more memory-efficient training

## References

- [PEFT Library](https://github.com/huggingface/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
