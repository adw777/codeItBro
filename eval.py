import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import GenerationConfig
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from statistics import mean
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvalMetrics:
    """Class to store evaluation metrics."""
    rouge1: float
    rouge2: float
    rougeL: float
    bleu: float
    bertscore_f1: float
    perplexity: float

def load_test_data(file_path: str) -> Dataset:
    """Load test dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

def prepare_model(model_path: str, max_length: int = 4096):
    """Load and prepare model for evaluation."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_length,
        dtype=None,
        load_in_4bit=True
    )
    
    generation_config = GenerationConfig(
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    return model, tokenizer, generation_config

def generate_response(model, tokenizer, generation_config, prompt: str) -> str:
    """Generate response for a given prompt."""
    inputs = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt}
    ], tokenize=False)
    
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids.cuda()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    response = response.split("assistant")[-1].strip()
    return response

def calculate_metrics(predictions: List[str], references: List[str]) -> EvalMetrics:
    """Calculate various evaluation metrics."""
    # Initialize scorers
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bert_scorer = evaluate.load("bertscore")
    perplexity_metric = evaluate.load("perplexity", module_type="metric")
    
    # Calculate ROUGE scores
    rouge_scores = [rouge.score(ref, pred) for ref, pred in zip(references, predictions)]
    
    # Calculate BLEU scores with smoothing
    smooth = SmoothingFunction()
    bleu_scores = []
    for ref, pred in zip(references, predictions):
        ref_tokens = [ref.split()]
        pred_tokens = pred.split()
        bleu_scores.append(sentence_bleu(ref_tokens, pred_tokens, 
                                       smoothing_function=smooth.method1))
    
    # Calculate BERTScore
    bert_results = bert_scorer.compute(
        predictions=predictions,
        references=references,
        lang="en"
    )
    
    # Calculate perplexity
    try:
        perplexity = perplexity_metric.compute(
            predictions=predictions,
            model_id="gpt2"
        )["mean_perplexity"]
    except:
        perplexity = float('inf')
    
    return EvalMetrics(
        rouge1=mean(score['rouge1'].fmeasure for score in rouge_scores),
        rouge2=mean(score['rouge2'].fmeasure for score in rouge_scores),
        rougeL=mean(score['rougeL'].fmeasure for score in rouge_scores),
        bleu=mean(bleu_scores),
        bertscore_f1=mean(bert_results['f1']),
        perplexity=perplexity
    )

def generate_test_examples(num_examples: int = 50) -> List[Dict[str, Any]]:
    """Generate diverse test examples."""
    test_examples = []
    
    # Example templates for different types of prompts
    templates = [
        # Problem-solving examples
        {"input": "Solve this programming problem: {}", "type": "problem_solving"},
        {"input": "How would you approach this situation: {}", "type": "analysis"},
        {"input": "Write a function to {}", "type": "coding"},
        
        # Creative writing examples
        {"input": "Write a short story about {}", "type": "creative"},
        {"input": "Describe a scene where {}", "type": "descriptive"},
        
        # Analytical examples
        {"input": "Analyze the implications of {}", "type": "analysis"},
        {"input": "Compare and contrast {} and {}", "type": "comparison"},
        
        # Technical examples
        {"input": "Explain how {} works", "type": "technical"},
        {"input": "What are the key components of {}", "type": "explanation"},
    ]
    
    # Example scenarios for each template type
    scenarios = {
        "problem_solving": [
            "implementing a binary search tree",
            "optimizing a database query",
            "handling concurrent user requests",
        ],
        "analysis": [
            "migrating a monolithic application to microservices",
            "choosing between SQL and NoSQL databases",
            "implementing OAuth authentication",
        ],
        "coding": [
            "sort an array using merge sort",
            "implement a cache with LRU policy",
            "create a simple pub/sub system",
        ],
        "creative": [
            "a successful software launch",
            "a day in the life of a programmer",
            "an AI breakthrough",
        ],
        "technical": [
            "Docker containerization",
            "blockchain consensus mechanisms",
            "neural network architectures",
        ],
    }
    
    # Generate diverse test examples
    for _ in range(num_examples):
        template = random.choice(templates)
        scenario = random.choice(scenarios.get(template["type"], scenarios["technical"]))
        
        test_example = {
            "input": template["input"].format(scenario),
            "type": template["type"]
        }
        test_examples.append(test_example)
    
    return test_examples

def main():
    # Paths to models
    original_model_path = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    finetuned_model_path = "finetuned_cot_model"
    
    # Generate or load test data
    test_data = generate_test_examples(num_examples=50)
    with open("test_data.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    # Initialize models
    logger.info("Loading original model...")
    original_model, original_tokenizer, generation_config = prepare_model(original_model_path)
    
    logger.info("Loading finetuned model...")
    finetuned_model, finetuned_tokenizer, _ = prepare_model(finetuned_model_path)
    
    # Generate predictions
    logger.info("Generating predictions...")
    original_predictions = []
    finetuned_predictions = []
    
    for example in tqdm(test_data):
        prompt = example["input"]
        
        # Generate responses from both models
        original_response = generate_response(
            original_model, original_tokenizer, generation_config, prompt
        )
        finetuned_response = generate_response(
            finetuned_model, finetuned_tokenizer, generation_config, prompt
        )
        
        original_predictions.append(original_response)
        finetuned_predictions.append(finetuned_response)
        
        # Save responses for manual inspection
        example["original_response"] = original_response
        example["finetuned_response"] = finetuned_response
    
    # Save all responses
    with open("evaluation_results.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    # Calculate metrics (using finetuned responses as reference)
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(original_predictions, finetuned_predictions)
    
    # Print results
    logger.info("\nEvaluation Results:")
    logger.info(f"ROUGE-1: {metrics.rouge1:.4f}")
    logger.info(f"ROUGE-2: {metrics.rouge2:.4f}")
    logger.info(f"ROUGE-L: {metrics.rougeL:.4f}")
    logger.info(f"BLEU: {metrics.bleu:.4f}")
    logger.info(f"BERTScore F1: {metrics.bertscore_f1:.4f}")
    logger.info(f"Perplexity: {metrics.perplexity:.4f}")

if __name__ == "__main__":
    main()