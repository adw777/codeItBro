import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import logging
import numpy as np
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_mixed_dataset(json_file: str) -> Dataset:
    """
    Load and prepare dataset with mixed formats (with/without CoT).
    Handles three possible formats:
    1. {"input": text, "cot": text, "solution": text}
    2. {"input": text, "response": text}
    3. {"input": text, "cot": text, "response": text}
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prepared_data = []
    for item in data:
        if "solution" in item:  # Format 1
            conversation = [
                {"role": "user", "content": item["input"]},
                {"role": "assistant", "content": f"Let me think about this step by step:\n{item['cot']}\n\nTherefore, here's the solution:\n{item['solution']}"}
            ]
        elif "cot" in item and "response" in item:  # Format 3
            conversation = [
                {"role": "user", "content": item["input"]},
                {"role": "assistant", "content": f"Let me think about this step by step:\n{item['cot']}\n\nHere's my response:\n{item['response']}"}
            ]
        else:  # Format 2 (no CoT)
            conversation = [
                {"role": "user", "content": item["input"]},
                {"role": "assistant", "content": item["response"]}
            ]
        prepared_data.append({"conversations": conversation})
    
    dataset = Dataset.from_list(prepared_data)
    logger.info(f"Loaded dataset with {len(dataset)} examples")
    return dataset

def prepare_model_and_tokenizer(max_seq_length: int = 4096):
    """Initialize the model and tokenizer with optimized settings for 48GB VRAM."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True
    )
    
    # Optimize LoRA parameters for better performance
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Increased rank for better capacity
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,  # Increased alpha for stronger adaptations
        lora_dropout=0.05,  # Added small dropout for regularization
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    
    return model, tokenizer

def prepare_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """Prepare dataset with efficient formatting."""
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(
            convo, 
            tokenize=False,
            add_generation_prompt=False
        ) for convo in convos]
        return {"text": texts}
    
    return dataset.map(
        formatting_prompts_func,
        batched=True,
        num_proc=4,  # Parallel processing
        remove_columns=dataset.column_names
    )

def main():
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster training
    torch.backends.cudnn.allow_tf32 = True
    
    logger.info("Loading dataset...")
    dataset = load_mixed_dataset("data.json")
    
    logger.info("Initializing model and tokenizer...")
    model, tokenizer = prepare_model_and_tokenizer()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Preparing dataset...")
    dataset = prepare_dataset(dataset, tokenizer)
    
    # Optimized training arguments for 2000+ examples on 48GB VRAM
    total_examples = 2840
    batch_size = 8  # Increased batch size since we have enough VRAM
    num_epochs = 3  # Multiple epochs for better learning
    
    # Calculate optimal steps
    steps_per_epoch = total_examples // (batch_size * 4)  # 4 is gradient_accumulation_steps
    total_steps = steps_per_epoch * num_epochs
    
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        max_steps=total_steps,
        learning_rate=3e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine_with_restarts",
        seed=42,
        output_dir="outputs",
        report_to="none",
        save_strategy="steps",
        save_steps=steps_per_epoch // 2,  # Save twice per epoch
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=4096,  # Increased sequence length
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt"
        ),
        dataset_num_proc=4,
        packing=True,  # Enable packing for more efficient training
        args=training_args,
    )
    
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    
    logger.info("Starting training...")
    trainer_stats = trainer.train()
    
    logger.info("Saving model...")
    model.save_pretrained("finetuned_cot_model")
    tokenizer.save_pretrained("finetuned_cot_model")
    
    logger.info(f"Training complete! Time taken: {trainer_stats.metrics['train_runtime']} seconds")
    logger.info(f"Final loss: {trainer_stats.metrics['train_loss']}")

if __name__ == "__main__":
    main()