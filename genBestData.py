import json
import time
import random
from typing import Dict, List, Any
import requests
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3
import re

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from dotenv import load_dotenv

load_dotenv()

MODEL_URL = os.getenv("MODEL_URL", "url")
MODEL_PATH = os.getenv("MODEL_PATH", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "your_default_auth_token")

def clean_json_response(response: str) -> str:
    """Clean the model's response to ensure valid JSON."""
    try:
        # Find the first { and last }
        start = response.find('{')
        end = response.rfind('}')
        
        if start == -1 or end == -1:
            return None
        
        # Extract the JSON part
        json_str = response[start:end+1]
        
        # Remove any markdown code block markers
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*', '', json_str)
        
        # Clean up any escaped quotes
        json_str = json_str.replace('\\"', '"')
        
        # Test if it's valid JSON
        json.loads(json_str)
        return json_str
    except:
        return None

def generate_question_prompt() -> List[Dict[str, str]]:
    """Generate a prompt for creating a complex Solidity contract question."""
    
    examples = [
        "Create a staking contract where users can stake tokens for different time periods and earn rewards based on the staking duration.",
        "Implement a vesting contract with a cliff period of 6 months and linear vesting over 2 years with monthly releases.",
        "Design a multi-token liquidity pool with dynamic fees based on pool volatility."
    ]
    
    system_message = """You are an expert Solidity developer creating complex smart contract challenges.
    Generate ONLY the question in valid JSON format with a single 'question' field."""
    
    user_message = f"""Create a detailed, practical Solidity smart contract challenge.

Example types of questions:
{examples[0]}
{examples[1]}
{examples[2]}

Your response must be in this EXACT format:
{{
    "question": "your detailed challenge here"
}}"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def model_run(messages: List[Dict[str, str]], max_tokens: int = 1000, max_retries: int = 3) -> str:
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
                "temperature": 0.9,
                "max_tokens": max_tokens,
                "top_p": 0.95,
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
                content = result['choices'][0]['message']['content']
                return content
            
            time.sleep(2 * (attempt + 1))  # Simple backoff

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            
    return None

def generate_single_question() -> Dict:
    """Generate a single complex question."""
    messages = generate_question_prompt()
    try:
        response = model_run(messages)
        if response:
            # Clean the response before parsing
            cleaned_response = clean_json_response(response)
            if cleaned_response:
                question_data = json.loads(cleaned_response)
                if "question" in question_data:
                    return {"question": question_data["question"].strip()}
    except Exception as e:
        print(f"Error in question generation: {str(e)}")
    
    return None

def generate_questions(num_questions: int = 1000, 
                      output_file: str = 'bestData.json',
                      max_workers: int = 2):
    """Generate multiple complex Solidity questions."""
    
    questions = []
    successful = 0
    failed = 0
    
    # Initialize file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=2)
    
    print(f"Generating {num_questions} Solidity contract questions...")
    
    with tqdm(total=num_questions, desc="Generating questions") as pbar:
        while successful < num_questions:
            batch_size = min(10, num_questions - successful)  # Process in small batches
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(generate_single_question) for _ in range(batch_size)]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            questions.append(result)
                            successful += 1
                            # Save progress
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(questions, f, indent=2, ensure_ascii=False)
                        else:
                            failed += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing question: {str(e)}")
                        failed += 1
                        pbar.update(1)
                    
                    # Small delay between questions
                    time.sleep(1)
            
            # Progress update
            if successful % 10 == 0:
                print(f"\nProgress: {successful}/{num_questions} questions generated")
                print(f"Failed attempts: {failed}")
    
    print(f"\nGeneration complete!")
    print(f"Successfully generated: {successful} questions")
    print(f"Failed attempts: {failed}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    generate_questions(
        num_questions=1000,
        output_file='bestData.json',
        max_workers=2
    )