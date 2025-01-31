import json
import time
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from threading import Lock
from typing import Dict, List, Any, Union

# Constants that would typically be in .env
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "openSource")
MODEL_URL = os.getenv("MODEL_URL", "url")  # Replace with actual URL
MODEL_PATH = os.getenv("MODEL_PATH", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")  # Replace with actual model name
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "your_default_auth_token")  # Replace with actual API key


def generate_prompt(question: str) -> List[Dict[str, str]]:
    """Generate messages for the model prompt."""
    system_message = """You are helping a programmer rephrase their coding question. Make it sound natural and conversational, 
    like how a developer would actually ask for help from an LLM. The rephrased question should be informal but clear."""
    
    user_message = f"""Rephrase this coding question to be more detailed and conversational by:
    1. Starting with a natural opener like "Hey, I'm trying to..." or "I need help with..."
    2. Adding context about why they need help or what they're working on
    3. Including what they've considered or what they're stuck on
    4. Mentioning specific test cases or examples they've tried
    5. Asking about edge cases or potential issues to watch out for
    6. Suggesting preferred programming languages, but keeping it flexible
    7. Keeping the [Topic: X] tag at the end

    Original question: {question}

    Make it sound like a real developer asking for help - casual but specific. Don't use formal headers, just make it flow naturally."""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def process_single_question(item: Dict[str, str], index: int, file_lock: Lock, output_file: str) -> tuple:
    """Process a single question using the model_run function."""
    original_question = item['input']
    messages = generate_prompt(original_question)
    
    try:
        rephrased_question = model_run(
            messages=messages,
            max_tokens=1000,
            llm_model="openSource",  # Since we're using the open source model
            max_retries=3,
            base_delay=5.0
        )
        
        if rephrased_question and not rephrased_question.startswith("Error:"):
            result = {
                "input": rephrased_question.strip()
            }
        else:
            result = item
            print(f"Warning: Keeping original question {index} due to API error: {rephrased_question}")
    except Exception as e:
        print(f"Error processing question {index}: {str(e)}")
        result = item
    
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
                while len(data) < index:
                    data.append(None)  # Fill gaps with None
                data.append(result)

            # Save updated data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving progress for question {index}: {str(e)}")
    
    return index, result

def rephrase_questions(input_file: str = 'data/cleaned_codingQues.json', 
                      output_file: str = 'data/updated_codingQues.json', 
                      max_workers: int = 3) -> None:
    """Main function to process and rephrase questions."""
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
                try:
                    index, result = future.result()
                    results[index] = result
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing future: {str(e)}")
        
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


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 32.0) -> float:
    """
    Calculate the delay for exponential backoff with jitter.
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = delay * 0.1 * random.random()
    return delay + jitter

def handle_retry_or_fail(
    attempt: int,
    max_retries: int,
    base_delay: float,
    error_msg: str
) -> Union[None, str]:
    """
    Helper to decide if we should retry or return an error.

    Args:
        attempt (int): Current attempt index (0-based).
        max_retries (int): Total allowed retries.
        base_delay (float): Base delay for exponential backoff.
        error_msg (str): Error message to return if no more retries.

    Returns:
        Union[None, str]: None if we will retry, or an error message if retries are exhausted.
    """
    if attempt < max_retries - 1:
        delay = exponential_backoff(attempt, base_delay)
        logger.info(f"{error_msg} - Retrying in {delay:.2f} seconds...")
        time.sleep(delay)
        return None
    else:
        logger.error(f"{error_msg} - Max retries reached.")
        return f"Error: {error_msg}"

def validate_messages(messages: Union[List[Dict[str, str]], Dict[str, str]]) -> Union[List[Dict[str, str]], str]:
    """
    Validate the 'messages' parameter, ensuring it's a list of dicts with 'role' and 'content'.
    If invalid, returns an error string. Otherwise returns the validated list.
    """
    if isinstance(messages, dict):
        messages = [messages]

    if not isinstance(messages, list) or len(messages) == 0:
        return "Error: 'messages' must be a non-empty list or a single dict."

    for msg in messages:
        if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
            return "Error: Each message must be a dict with 'role' and 'content'."

    return messages

def model_run(
    messages: Union[List[Dict[str, str]], Dict[str, str]],
    max_tokens: int,
    llm_model: str = LLM_MODEL,
    max_retries: int = 3,
    base_delay: float = 5.0
) -> str:
    """
    Runs the specified language model (GPT-4 or open-source) and returns the assistant's message.

    Args:
        messages (List[Dict[str, str]]): The input messages for the model.
        llmModel (str): The model type ("gpt-4" or "openSource"). Defaults to the value from environment variables.
        max_tokens (int): The maximum number of tokens to generate. Defaults to 4096.
        max_retries (int): The maximum number of retry attempts. Defaults to 3.
        base_delay (float): The base delay for exponential backoff. Defaults to 1.0.
    Returns:
        str: The assistant's message generated by the model.
    """
    validated = validate_messages(messages)
    if isinstance(validated, str):
        logger.error(validated)
        return validated
    else:
        messages = validated

    for attempt in range(max_retries):
        try:
            if llm_model != "openSource":
                logger.error(f"Unsupported model type: {llm_model}")
                return "Error: Unsupported model type."

            url = MODEL_URL
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AUTH_TOKEN}"
                }
            data: Dict[str, Any] = {
                "model": MODEL_PATH,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": max_tokens,
                "top_p": 0.9,
            }

            response: Response = requests.post(url, headers=headers, json=data, verify=False, timeout=600)

            if response.status_code in [400, 429]:
                code_name = "Bad Request" if response.status_code == 400 else "Rate Limit"
                logger.warning(f"Attempt {attempt+1}/{max_retries}: HTTP {response.status_code} ({code_name}).")
                try:
                    err_details = response.json()
                    logger.warning(f"Server response: {err_details}")
                except Exception:
                    logger.warning("Unable to parse server error JSON.")

                retry_decision = handle_retry_or_fail(
                    attempt,
                    max_retries,
                    base_delay,
                    f"HTTP {response.status_code} {code_name}"
                )
                if retry_decision is not None:
                    return retry_decision
                else:
                    continue  

            if not response.ok:
                logger.warning(f"Attempt {attempt+1}/{max_retries}: HTTP {response.status_code}.")
                retry_decision = handle_retry_or_fail(
                    attempt,
                    max_retries,
                    base_delay,
                    f"HTTP {response.status_code} Error"
                )
                if retry_decision is not None:
                    return retry_decision
                else:
                    continue

            try:
                result = response.json()
            except Exception as parse_exc:
                logger.error(f"JSON parse error on attempt {attempt+1}: {parse_exc}", exc_info=True)
                retry_decision = handle_retry_or_fail(
                    attempt,
                    max_retries,
                    base_delay,
                    "Invalid JSON in response"
                )
                if retry_decision is not None:
                    return retry_decision
                else:
                    continue

            if 'choices' not in result or not result['choices']:
                logger.error("No 'choices' found in response.")
                retry_decision = handle_retry_or_fail(
                    attempt,
                    max_retries,
                    base_delay,
                    "Missing 'choices' in response"
                )
                if retry_decision is not None:
                    return retry_decision
                else:
                    continue

            return result['choices'][0]['message']['content']

        except requests.exceptions.RequestException as net_err:
            logger.error(f"Network error on attempt {attempt+1}: {net_err}")
            retry_decision = handle_retry_or_fail(
                attempt,
                max_retries,
                base_delay,
                f"Network error: {net_err}"
            )
            if retry_decision is not None:
                return retry_decision
            else:
                continue

        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt+1}: {e}", exc_info=True)
            retry_decision = handle_retry_or_fail(
                attempt,
                max_retries,
                base_delay,
                f"Unexpected error: {e}"
            )
            if retry_decision is not None:
                return retry_decision
            else:
                continue

    return "Error: All retry attempts failed."


if __name__ == "__main__":
    rephrase_questions(max_workers=8)