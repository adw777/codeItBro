import json
import time
import random
from typing import Dict, List, Any, Union
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from threading import Lock

# Constants from environment variables
from dotenv import load_dotenv

load_dotenv()

MODEL_URL = os.getenv("MODEL_URL", "url")  # Replace with actual URL
MODEL_PATH = os.getenv("MODEL_PATH", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "your_default_auth_token")

def generate_variation_prompt(question: str, answer: str) -> List[Dict[str, str]]:
    """Generate a prompt to create variations of a question-answer pair."""
    system_message = """You are an expert in blockchain and Solidity programming. Generate a new question-answer pair 
    related to the original one, but with a different perspective or scenario. Keep the technical accuracy but vary the 
    context, complexity, or specific implementation details."""
    
    user_message = f"""Given this original Q&A pair about Ethereum/Solidity development:

Original Question: {question}
Original Answer: {answer}

Create a new, different but related question-answer pair that:
1. Covers similar concepts but from a different angle
2. Uses different examples or scenarios
3. Maintains technical accuracy
4. Has similar depth and detail level
5. Feels natural and practical

Respond in strict JSON format like this:
{{
    "question": "your new question here",
    "answer": "your new answer here"
}}"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 32.0) -> float:
    """Calculate delay for exponential backoff with jitter."""
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = delay * 0.1 * random.random()
    return delay + jitter

def model_run(messages: List[Dict[str, str]], max_tokens: int = 1000, max_retries: int = 3, base_delay: float = 5.0) -> str:
    """Run the model with retry logic."""
    for attempt in range(max_retries):
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AUTH_TOKEN}"
            }
            
            data = {
                "model": MODEL_PATH,
                "messages": messages,
                "temperature": 0.7,  # Higher temperature for more variation
                "max_tokens": max_tokens,
                "top_p": 0.9,
            }

            response = requests.post(
                MODEL_URL,
                headers=headers,
                json=data,
                verify=False,
                timeout=600
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            
            delay = exponential_backoff(attempt, base_delay)
            print(f"Attempt {attempt + 1} failed. Retrying in {delay:.2f} seconds...")
            time.sleep(delay)

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                delay = exponential_backoff(attempt, base_delay)
                time.sleep(delay)
            else:
                return f"Error: {str(e)}"

    return "Error: All retry attempts failed."

def process_qa_pair(qa_pair: Dict[str, str], index: int, variations_per_pair: int = 5) -> List[Dict[str, str]]:
    """Process a single Q&A pair and generate variations."""
    variations = []
    original_question = qa_pair['question']
    original_answer = qa_pair['answer']
    
    for i in range(variations_per_pair):
        messages = generate_variation_prompt(original_question, original_answer)
        try:
            response = model_run(messages)
            if not response.startswith("Error"):
                try:
                    variation = json.loads(response)
                    variations.append(variation)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON for pair {index}, variation {i}")
                    continue
        except Exception as e:
            print(f"Error processing pair {index}, variation {i}: {str(e)}")
            continue
            
        # Add small delay between variations to avoid rate limiting
        time.sleep(1)
    
    return variations

def extend_dataset(input_file: str = 'EVM.json', 
                  output_file: str = 'EVMextended.json',
                  variations_per_pair: int = 5,
                  max_workers: int = 3):
    """Main function to extend the dataset with variations."""
    try:
        # Read input data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        extended_data = []
        # Add original pairs first
        extended_data.extend(data)
        
        # Process Q&A pairs with progress bar
        with tqdm(total=len(data), desc="Generating variations") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(process_qa_pair, item, idx, variations_per_pair): idx 
                    for idx, item in enumerate(data)
                }
                
                for future in as_completed(future_to_index):
                    variations = future.result()
                    extended_data.extend(variations)
                    pbar.update(1)
                    
                    # Save progress periodically
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(extended_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nOriginal pairs: {len(data)}")
        print(f"Total pairs after extension: {len(extended_data)}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    extend_dataset(
        input_file='EVM.json',
        output_file='EVMextended.json',
        variations_per_pair=5,
        max_workers=3
    )