#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Language Models
Supports JSONL datasets with chat message format
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="LoRA fine-tuning script")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="Model name or path (default: 'Qwen/Qwen2.5-Coder-0.5B-Instruct')"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to JSONL dataset file"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder for trained model"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (r parameter)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization (QLoRA)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps"
    )
    
    return parser.parse_args()


def load_jsonl_dataset(file_path):
    """Load JSONL dataset with messages format"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


def tokenize_function(entry, tokenizer, max_length=512):
    # 1. Get the full prompt string
    prompt = tokenizer.apply_chat_template(entry["messages"], tokenize=False)
    # 2. Tokenize the full prompt
    tokenized = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    # 3. Copy input_ids to labels
    labels = tokenized["input_ids"].copy()
    
    # 4. Find where "<|im_start|>assistant" begins
    assistant_start_text = "<|im_start|>assistant"
    assistant_start_ids = tokenizer.encode(assistant_start_text, add_special_tokens=False)
    
    # Find the position of assistant start in the tokenized sequence
    assistant_start_pos = None
    for i in range(len(labels) - len(assistant_start_ids) + 1):
        if labels[i:i+len(assistant_start_ids)] == assistant_start_ids:
            assistant_start_pos = i
            break
    
    # 5. Mask everything before assistant start
    if assistant_start_pos is not None:
        for i in range(assistant_start_pos):
            labels[i] = -100
            # tokenized["attention_mask"][i] = 0
    else:
        # No assistant found, mask everything
        for i in range(len(labels)):
            labels[i] = -100
            # tokenized["attention_mask"][i] = 0
    
    # 6. Mask padding tokens (where attention_mask is 0)
    for i, mask_val in enumerate(tokenized["attention_mask"]):
        if mask_val == 0:
            labels[i] = -100
    
    # -100 is the ignore index in PyTorch's CrossEntropyLoss
    tokenized["labels"] = labels
    return tokenized


def main():
    args = parse_args()
    
    print("=" * 60)
    print("LoRA Fine-tuning Configuration")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data Path: {args.data_path}")
    print(f"Output Folder: {args.output_folder}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"LoRA Rank: {args.lora_rank}")
    print(f"LoRA Alpha: {args.lora_alpha}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"4-bit Quantization: {args.use_4bit}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Load tokenizer
    print("\n📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print("🤖 Loading model...")
    # model_kwargs = {
    #     "trust_remote_code": True,
    #     "torch_dtype": torch.float16,
    #     "device_map": "auto",
    # }
    
    # if args.use_4bit:
    #     from transformers import BitsAndBytesConfig
    #     model_kwargs["quantization_config"] = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_compute_dtype=torch.float16,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_use_double_quant=True,
    #     )
    
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    # Prepare model for k-bit training if using quantization
    # if args.use_4bit:
    #     model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print("⚙️  Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    print("\n📊 Loading dataset...")
    messages = load_jsonl_dataset(args.data_path)
    print(f"Dataset size: {len(messages)} examples")
    
    # # Create dataset
    # dataset = Dataset.from_dict({'messages': messages})
    
    # Tokenize dataset
    print("🔤 Tokenizing dataset...")
    # tokenized_dataset = dataset.map(
    #     lambda x: tokenize_function(x, tokenizer, args.max_seq_length),
    #     batched=True,
    #     remove_columns=['messages'],
    #     desc="Tokenizing data"
    # )
    tokenized_dataset = [tokenize_function(entry, tokenizer, args.max_seq_length) for entry in messages]
    
    print(f"Training examples: {len(tokenized_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_folder,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none"
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Initialize trainer
    print("\n🚀 Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    print("\n🏋️  Starting training...")
    trainer.train()
    
    # Save final model
    print("\n💾 Saving model...")
    model.save_pretrained(args.output_folder)
    tokenizer.save_pretrained(args.output_folder)
    
    print(f"\n✅ Training complete! Model saved to {args.output_folder}")
    
    # Save training configuration
    config_path = os.path.join(args.output_folder, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Training configuration saved to {config_path}")


if __name__ == "__main__":
    main()
