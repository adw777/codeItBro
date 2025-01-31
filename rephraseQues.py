import json
import requests
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from threading import Lock

def generate_ollama_prompt(question):
    return f"""You are helping a programmer rephrase their coding question. Make it sound natural and conversational, like how a developer would actually ask for help from an LLM. The rephrased question should be informal but clear.

    Original question: {question}

    Make it more detailed but conversational by:
    1. Starting with a natural opener like "Hey, I'm trying to..." or "I need help with..."
    2. Adding context about why they need help or what they're working on
    3. Including what they've considered or what they're stuck on
    4. Mentioning specific test cases or examples they've tried
    5. Asking about edge cases or potential issues to watch out for
    6. Suggesting preferred programming languages, but keeping it flexible
    7. Keeping the [Topic: X] tag at the end

    Make it sound like a real developer asking for help - casual but specific. Drop any formal structure like "Requirements:" or "Example Test Cases:" and instead weave these details naturally into the conversation.

    Remember to:
    - Keep it conversational and informal
    - Make it feel like a genuine question someone would ask
    - Include technical details but in a natural way
    - Maintain the original problem's complexity
    - Keep the core challenge clear
    - Do not include headers like "Here's the rephrased question:"

    Write it as one natural, flowing question without formal headers or sections."""

def query_ollama(prompt, retries=3, delay=1):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama3.1:latest",
        "prompt": prompt,
        "stream": False
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                print(f"Error querying Ollama after {retries} attempts: {e}")
                return None
            time.sleep(delay * (attempt + 1))
    return None

def process_single_question(item, index, file_lock, output_file):
    original_question = item['input']
    prompt = generate_ollama_prompt(original_question)
    rephrased_question = query_ollama(prompt)
    
    if rephrased_question:
        result = {
            "input": rephrased_question.strip()
        }
    else:
        result = item
        print(f"Warning: Keeping original question {index} due to API error")
    
    # Save progress to the main file with lock
    with file_lock:
        try:
            # Read existing data
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = []

            # Update or append the result
            if len(data) > index:
                data[index] = result
            else:
                data.append(result)

            # Save updated data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving progress for question {index}: {str(e)}")
    
    return index, result

def rephrase_questions(input_file='data/cleaned_codingQues.json', output_file='data/updated_codingQues.json', max_workers=3):
    try:
        # Create a new file or clear existing one
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
        
        # Read the cleaned JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Initialize progress bar
        total_questions = len(data)
        pbar = tqdm(total=total_questions, desc="Processing questions")
        
        # Initialize results list with None values
        results = [None] * total_questions
        
        # Create a lock for file access
        file_lock = Lock()
        
        # Process questions using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_question, item, i, file_lock, output_file): i 
                for i, item in enumerate(data)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_index):
                index, result = future.result()
                results[index] = result
                pbar.update(1)
        
        pbar.close()
        
        # Final verification of results
        with open(output_file, 'r', encoding='utf-8') as f:
            final_data = json.load(f)
            
        print(f"\nSuccessfully processed {len(final_data)} questions")
        print("\nSample of updated questions:")
        for item in final_data[:2]:
            print(item['input'])
            print("---")
            
    except FileNotFoundError:
        print(f"Error: Could not find the input file {input_file}")
    except json.JSONDecodeError:
        print(f"Error: The input file {input_file} is not valid JSON")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    rephrase_questions(max_workers=8)



# # for testing-- creates example.json

# import json
# import requests
# import time
# from tqdm import tqdm

# def generate_ollama_prompt(question):
#     return f"""You are helping a programmer rephrase their coding question. Make it sound natural and conversational, like how a developer would actually ask for help from an LLM. The rephrased question should be informal but clear.

# Original question: {question}

# Make it more detailed but conversational by:
# 1. Starting with a natural opener like "Hey, I'm trying to..." or "I need help with..."
# 2. Adding context about why they need help or what they're working on
# 3. Including what they've considered or what they're stuck on
# 4. Mentioning specific test cases or examples they've tried
# 5. Asking about edge cases or potential issues to watch out for
# 6. Suggesting preferred programming languages, but keeping it flexible
# 7. Keeping the [Topic: X] tag at the end

# Make it sound like a real developer asking for help - casual but specific. Drop any formal structure like "Requirements:" or "Example Test Cases:" and instead weave these details naturally into the conversation.

# Remember to:
# - Keep it conversational and informal
# - Make it feel like a genuine question someone would ask
# - Include technical details but in a natural way
# - Maintain the original problem's complexity
# - Keep the core challenge clear

# Write it as one natural, flowing question without formal headers or sections."""

# def query_ollama(prompt, retries=3, delay=1):
#     url = "http://localhost:11434/api/generate"
    
#     headers = {
#         "Content-Type": "application/json"
#     }
    
#     data = {
#         "model": "llama3.1:latest",
#         "prompt": prompt,
#         "stream": False
#     }
    
#     for attempt in range(retries):
#         try:
#             response = requests.post(url, headers=headers, json=data)
#             response.raise_for_status()
#             return response.json()['response']
#         except requests.exceptions.RequestException as e:
#             if attempt == retries - 1:
#                 print(f"Error querying Ollama after {retries} attempts: {e}")
#                 return None
#             time.sleep(delay * (attempt + 1))
    
#     return None

# def test_rephrase_questions(input_file='cleaned_codingQues.json', output_file='example.json', num_questions=2):
#     try:
#         # Read the cleaned JSON file
#         with open(input_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         # Take only the first num_questions
#         test_data = data[:num_questions]
        
#         updated_data = []
#         print(f"Processing {len(test_data)} questions as a test run...")
        
#         # Create a comparison dictionary to store before/after
#         comparison_data = []
        
#         # Process each question with progress bar
#         for item in tqdm(test_data):
#             original_question = item['input']
            
#             # Generate prompt for the question
#             prompt = generate_ollama_prompt(original_question)
            
#             # Get rephrased question from Ollama
#             rephrased_question = query_ollama(prompt)
            
#             if rephrased_question:
#                 # Store both original and rephrased for comparison
#                 comparison_data.append({
#                     "original": original_question,
#                     "rephrased": rephrased_question.strip()
#                 })
                
#                 updated_data.append({
#                     "input": rephrased_question.strip()
#                 })
#             else:
#                 print(f"Warning: Keeping original question due to API error: {original_question[:100]}...")
#                 comparison_data.append({
#                     "original": original_question,
#                     "rephrased": original_question  # Keep original if API fails
#                 })
#                 updated_data.append(item)
            
#             # Small delay to prevent overwhelming the API
#             time.sleep(0.1)
        
#         # Write the test results to example.json
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
#         print(f"\nTest run completed. Results saved to {output_file}")
#         print("\nSample comparison of original vs rephrased:")
#         for i, item in enumerate(comparison_data[:2], 1):
#             print(f"\nQuestion {i}:")
#             print("Original:", item['original'])
#             print("Rephrased:", item['rephrased'])
#             print("-" * 80)
            
#     except FileNotFoundError:
#         print(f"Error: Could not find the input file {input_file}")
#     except json.JSONDecodeError:
#         print(f"Error: The input file {input_file} is not valid JSON")
#     except Exception as e:
#         print(f"An unexpected error occurred: {str(e)}")

# if __name__ == "__main__":
#     test_rephrase_questions()