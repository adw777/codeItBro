import json
import requests
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from threading import Lock
import re
import argparse

def extract_cot_and_response(response: str) -> tuple:
    """
    Extract Chain of Thought (between <think></think> tags) and detailed response
    Returns tuple of (cot, response)
    """
    cot_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    cot = cot_match.group(1).strip() if cot_match else ""
    
    # Everything after </think> tag is the detailed response
    response_text = re.split(r'</think>\s*', response)[-1].strip()
    
    return cot, response_text

def generate_prompt(query: str) -> list:
    """Generate messages for the vLLM API with detailed requirements analysis prompt"""
    return [{
        "role": "user",
        "content": f"""As a senior Solidity architect, analyze the following smart contract development query and provide a detailed implementation plan. 

First, within <think></think> tags, perform a thorough analysis considering:
1. Core Requirements Analysis
   - Identify all primary and secondary features
   - Map out data structures and state variables
   - Define key interfaces and interactions
   - List all required user roles and permissions

2. Technical Considerations
   - Required smart contract standards (ERC20, ERC721, etc.)
   - Gas optimization opportunities
   - Storage vs. Memory trade-offs
   - External dependencies and integrations

3. Security Analysis
   - Potential attack vectors
   - Required security patterns
   - Access control requirements
   - Economic security considerations

4. Edge Cases and Challenges
   - Failure modes and recovery mechanisms
   - State transition edge cases
   - Resource limits and constraints
   - Cross-function interaction risks

Then, provide a detailed implementation plan including:
1. Smart Contract Architecture
   - Contract hierarchy and inheritance
   - Interface definitions
   - Library integrations
   - Storage layout

2. Implementation Roadmap
   - Core function specifications
   - Data structure implementations
   - Access control mechanisms
   - Event definitions

3. Code Examples
   - Key function implementations
   - Critical security checks
   - Important modifiers
   - Testing considerations

4. Deployment and Testing Plan
   - Contract deployment sequence
   - Configuration requirements
   - Test scenarios
   - Verification steps

Query to analyze: {query}"""
    }]

def query_vllm(prompt: list, retries: int = 3, delay: float = 1.0) -> str:
    """Query the vLLM API with retries"""
    url = "http://localhost:9000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "messages": prompt,
        "temperature": 0.3,  # Lower temperature for more focused responses
        "max_tokens": 4000   # Increased for detailed responses
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

def process_single_query(item: dict, index: int, file_lock: Lock, output_file: str) -> tuple:
    """Process a single query and save results"""
    try:
        # Generate prompt and get model response
        prompt = generate_prompt(item['query'])
        response = query_vllm(prompt)
        
        if response:
            # Extract CoT and response
            cot, detailed_response = extract_cot_and_response(response)
            
            result = {
                "query": item['query'],
                "cot": cot,
                "response": detailed_response
            }
        else:
            print(f"Warning: No response for query {index}")
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
                    while len(data) < index:
                        data.append(None)
                    data.append(result)
                
                # Save updated data
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                print(f"Error saving progress for query {index}: {str(e)}")
        
        return index, result
        
    except Exception as e:
        print(f"Error processing query {index}: {str(e)}")
        return index, None

def generate_requirements(input_file: str = 'solidityQueries.json',
                        output_file: str = 'solidityRequirements.json',
                        max_workers: int = 4,
                        test_mode: bool = False) -> None:
    """
    Main function to generate detailed requirements and plans from queries
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        max_workers: Number of parallel workers
        test_mode: If True, only process first 2 queries
    """
    try:
        # Create output file if it doesn't exist
        if not os.path.exists(output_file):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
        
        # Read input queries
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # If test mode, only take first 2 queries
        if test_mode:
            print("Running in test mode with first 2 queries...")
            data = data[:2]
        
        total_queries = len(data)
        print(f"Processing {total_queries} queries...")
        
        # Initialize progress bar
        pbar = tqdm(total=total_queries, desc="Generating requirements")
        
        # Create file access lock
        file_lock = Lock()
        
        # Process queries in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_query, item, i, file_lock, output_file): i
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
        
        print(f"\nSuccessfully processed {len(final_data)} queries")
        if test_mode:
            print("\nSample of generated requirements:")
            for item in final_data[:2]:
                if item:
                    print("\nQuery:", item['query'][:100], "...")
                    print("\nAnalysis:", item['cot'][:100], "...")
                    print("\nImplementation Plan:", item['response'][:100], "...")
                    print("-" * 80)
                
    except FileNotFoundError:
        print(f"Error: Could not find the input file {input_file}")
    except json.JSONDecodeError:
        print(f"Error: The input file {input_file} is not valid JSON")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate requirements and plans for Solidity queries')
    parser.add_argument('--test', action='store_true', help='Run in test mode with 2 queries')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()
    
    generate_requirements(
        max_workers=args.workers,
        test_mode=args.test
    )