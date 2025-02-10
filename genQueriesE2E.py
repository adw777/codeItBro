import json
import time
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from threading import Lock
import random
from typing import Dict, List, Any, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
MODEL_URL = os.getenv("MODEL_URL", "url")
MODEL_PATH = os.getenv("MODEL_PATH", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "your_default_auth_token")

def generate_prompt(index: int) -> List[Dict[str, str]]:
    """Generate messages for the model prompt."""
    system_message = """You are an experienced blockchain developer who specializes in creating complex Solidity smart contracts. 
    Your task is to generate detailed, realistic questions that a developer might ask when planning to build sophisticated smart contracts."""
    
    # Example scenarios to guide question generation
    examples = """Here are some example types of questions to guide your generation:

    1. "I'm building a DeFi protocol that needs to handle flash loans with dynamic interest rates based on utilization. Looking for guidance on implementing the core lending pool contract with focus on security against reentrancy and price manipulation. The contract should track lending positions using ERC721 tokens. Any suggestions on structuring this? [DeFi, Flash Loans]"

    2. "Need help designing a multi-sig wallet contract that supports both ERC20 and ERC721 tokens, with daily transaction limits and tiered approval requirements based on transaction value. Also want to add a recovery mechanism for lost keys. What's the best way to structure this? [Multi-sig, Security]"

    3. "Working on a complex staking mechanism where rewards are calculated based on multiple factors: time staked, amount staked, and platform activity. Users should be able to stake LP tokens and earn multiple reward tokens with different vesting schedules. How should I approach this? [Staking, DeFi]"

    4. "Building a decentralized marketplace for fractional real estate ownership. Need help with the tokenization contract that handles property share distribution, dividend payments, and voting rights. Looking for a secure way to manage property transfers and ensure compliance with ownership limits. [Real Estate, Tokenization]"

    Generate a unique, complex smart contract development question that:
    1. Focuses on real-world use cases
    2. Includes specific technical requirements
    3. Asks about security considerations
    4. Mentions relevant standards (ERC20, ERC721, etc.)
    5. Includes implementation challenges
    6. Tags the domain at the end [Domain1, Domain2]
    7. DO NOT INCLUDE ANY PREAMBLES LIKE:"Here's a complex smart contract development question for a Solidity smart contract development task:"
    8. Just give the query """

    user_message = f"""Generate question #{index} for a Solidity smart contract development task. 
    Make it detailed and specific, similar to the examples but completely different in use case and requirements. 
    Focus on complex DeFi, GameFi, governance, or enterprise use cases. 
    Include specific technical challenges and security considerations."""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": examples + "\n\n" + user_message}
    ]

def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 32.0) -> float:
    """Calculate delay for exponential backoff with jitter."""
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = delay * 0.1 * random.random()
    return delay + jitter

def model_run(messages: List[Dict[str, str]], max_retries: int = 3, base_delay: float = 5.0) -> str:
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
                "temperature": 0.9,  # Higher temperature for more diverse questions
                "max_tokens": 1000,
                "top_p": 0.95,
            }

            response = requests.post(MODEL_URL, headers=headers, json=data, verify=False, timeout=600)
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and result['choices']:
                return result['choices'][0]['message']['content']
            
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Error: {str(e)}"
            delay = exponential_backoff(attempt, base_delay)
            time.sleep(delay)
            continue
    
    return "Error: All retry attempts failed."

def process_single_query(index: int, file_lock: Lock, output_file: str) -> tuple:
    """Generate a single Solidity development query."""
    try:
        messages = generate_prompt(index)
        query = model_run(messages)
        
        if query and not query.startswith("Error:"):
            result = {
                "query": query.strip()
            }
        else:
            print(f"Warning: Error generating query {index}: {query}")
            return index, None
        
        # Save progress with lock
        with file_lock:
            try:
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    data = []
                
                if len(data) > index:
                    data[index] = result
                else:
                    while len(data) < index:
                        data.append(None)
                    data.append(result)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                print(f"Error saving progress for query {index}: {str(e)}")
        
        return index, result
        
    except Exception as e:
        print(f"Error processing query {index}: {str(e)}")
        return index, None

def generate_queries(num_queries: int = 2000,
                    output_file: str = 'solidityQueries.json',
                    max_workers: int = 4,
                    test_mode: bool = False) -> None:
    """
    Main function to generate Solidity development queries
    
    Args:
        num_queries: Number of queries to generate
        output_file: Output JSON file path
        max_workers: Number of parallel workers
        test_mode: If True, only generate 5 queries for testing
    """
    try:
        # Initialize output file
        if not os.path.exists(output_file):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
        
        # Adjust number of queries for test mode
        if test_mode:
            num_queries = 5
            print("Running in test mode with 5 queries...")
        
        print(f"Generating {num_queries} Solidity development queries...")
        
        # Initialize progress bar
        pbar = tqdm(total=num_queries, desc="Generating queries")
        
        # Create file access lock
        file_lock = Lock()
        
        # Process queries in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_query, i, file_lock, output_file): i
                for i in range(num_queries)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_index):
                try:
                    index, result = future.result()
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing future: {str(e)}")
        
        pbar.close()
        
        # Verify final results
        with open(output_file, 'r', encoding='utf-8') as f:
            final_data = json.load(f)
        
        print(f"\nSuccessfully generated {len(final_data)} queries")
        if test_mode:
            print("\nSample of generated queries:")
            for item in final_data[:2]:
                if item:
                    print("\nQuery:", item['query'])
                    print("-" * 80)
                
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Solidity development queries')
    parser.add_argument('--test', action='store_true', help='Run in test mode with 5 queries')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--num_queries', type=int, default=2000, help='Number of queries to generate')
    args = parser.parse_args()
    
    generate_queries(
        num_queries=args.num_queries,
        max_workers=args.workers,
        test_mode=args.test
    )