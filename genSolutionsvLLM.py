import json
import requests
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from threading import Lock
import re
import argparse

def convert_sets_to_lists(obj):
    """Recursively convert sets to lists in nested data structures"""
    if isinstance(obj, dict):
        return {key: convert_sets_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)  # Convert any other types to strings

def extract_cot_and_solution(response: str) -> tuple:
    """
    Extract Chain of Thought (between <think></think> tags) and solution from model response.
    Returns tuple of (cot, solution)
    """
    # Extract CoT between <think> tags
    cot_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    cot = cot_match.group(1).strip() if cot_match else ""
    
    # Everything after </think> tag is considered the solution
    solution = re.split(r'</think>\s*', response)[-1].strip()
    
    return cot, solution

def generate_vllm_prompt(question: str) -> list:
    """Generate messages for the vLLM API"""
    return [{
        "role": "user",
        "content": f"""
        You are a principal Solidity engineer, an expert in blockchain development and smart contract security. Given a Solidity smart contract problem, provide a detailed, secure, and optimized solution.

        First, analyze the problem step by step within <think></think> tags, considering:
        - The core functionality and required features.
        - Best practices for security, gas efficiency, and maintainability.
        - Potential attack vectors (e.g., reentrancy, frontrunning, overflow/underflow).
        - Compliance with standards (e.g., OpenZeppelin, ERC-20, ERC-721, ERC-1155, EIP-2535).
        - Upgradeability and extensibility.

        Then, provide a clean, well-structured Solidity implementation:
        - Use modular and reusable contract architecture.
        - Apply appropriate access controls and role-based permissions.
        - Follow Solidity best practices for versioning, error handling, and state management.
        - Implement and explain any necessary optimizations.

        Ensure the solution includes:
        - Detailed comments for clarity.
        - Unit test considerations.
        - Edge case handling.
        - Gas-efficient implementations.

        Give your best!

        Question: {question}"""
    }]

def query_vllm(prompt: list, retries: int = 3, delay: float = 1.0) -> str:
    """Query the vLLM API with retries"""
    url = "http://localhost:9000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "messages": prompt
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                print(f"Error querying vLLM after {retries} attempts: {e}")
                return None
            time.sleep(delay * (attempt + 1))
    return None

def process_single_question(item: dict, index: int, file_lock: Lock, output_file: str) -> tuple:
    """Process a single question and save results"""
    try:
        # Generate prompt and get model response
        prompt = generate_vllm_prompt(item['question'])
        response = query_vllm(prompt)
        
        if response:
            # Extract CoT and solution
            cot, solution = extract_cot_and_solution(response)
            
            # Create result dictionary and convert any sets to lists
            result = convert_sets_to_lists({
                "question": item['question'],
                "cot": cot,
                "solution": solution
            })
        else:
            print(f"Warning: No response for question {index}")
            return index, None
        
        # Save progress to file with lock
        with file_lock:
            try:
                # Read existing data
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    data = []
                
                # Update or append result
                if len(data) > index:
                    data[index] = result
                else:
                    # Fill gaps with None
                    while len(data) < index:
                        data.append(None)
                    data.append(result)
                
                # Save updated data
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                print(f"Error saving progress for question {index}: {str(e)}")
        
        return index, result
        
    except Exception as e:
        print(f"Error processing question {index}: {str(e)}")
        return index, None

def generate_solutions(input_file: str = 'bestDataNew.json',
                      output_file: str = 'soliditySols.json',
                      max_workers: int = 4,
                      test_mode: bool = False) -> None:
    """
    Main function to generate solutions from coding questions
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        max_workers: Number of parallel workers
        test_mode: If True, only process first 2 questions
    """
    try:
        # Create output file if it doesn't exist
        if not os.path.exists(output_file):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
        
        # Read input questions
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # If test mode, only take first 2 questions
        if test_mode:
            print("Running in test mode with first 2 questions...")
            data = data[:2]
        
        total_questions = len(data)
        print(f"Processing {total_questions} questions...")
        
        # Initialize progress bar
        pbar = tqdm(total=total_questions, desc="Generating solutions")
        
        # Create file access lock
        file_lock = Lock()
        
        # Process questions in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_question, item, i, file_lock, output_file): i
                for i, item in enumerate(data)
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
        
        print(f"\nSuccessfully processed {len(final_data)} questions")
        if test_mode:
            print("\nSample of generated solutions:")
            for item in final_data[:2]:
                if item:  # Check if item exists
                    print("\nQuestion:", item['question'][:100], "...")
                    print("\nCoT:", item['cot'][:100], "...")
                    print("\nSolution:", item['solution'][:100], "...")
                    print("-" * 80)
                
    except FileNotFoundError:
        print(f"Error: Could not find the input file {input_file}")
    except json.JSONDecodeError:
        print(f"Error: The input file {input_file} is not valid JSON")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate solutions for coding questions')
    parser.add_argument('--test', action='store_true', help='Run in test mode with 2 questions')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()
    
    generate_solutions(
        max_workers=args.workers,
        test_mode=args.test
    )