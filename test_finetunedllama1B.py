from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import TextStreamer
import json
from datetime import datetime

def load_model():
    print("Loading coding model and tokenizer...")
    # Load tokenizer from local directory
    tokenizer = AutoTokenizer.from_pretrained("finetuned_coding_model")
    
    # Load model from local directory
    model = AutoModelForCausalLM.from_pretrained(
        "finetuned_coding_model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def generate_code_solution(model, tokenizer, coding_question):
    # Create prompt
    prompt = f"""
Question: {coding_question}
Write a Python solution for this problem:
"""
    
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"\nGenerating solution for: {coding_question}")
    print("-" * 50)
    
    # Generate text
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,  # Reduced as code solutions are typically shorter
        temperature=0.2,     # Lower temperature for more focused code generation
        top_p=0.95,
        repetition_penalty=1.2,
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the generated text
    response = generated_text[len(prompt):].strip()
    
    # Print the output
    print(response)
    print("\n" + "=" * 50)
    
    return {
        "question": coding_question,
        "solution": response
    }

def save_to_json(solutions, filename=None):
    # Generate filename with timestamp if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_solutions_{timestamp}.json"
    
    # Save solutions to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(solutions, f, indent=2, ensure_ascii=False)
    
    print(f"\nSolutions saved to {filename}")

def main():
    # Free memory
    torch.cuda.empty_cache()
    
    # Load model
    model, tokenizer = load_model()
    
    # Test coding questions (basic programming problems)
    test_questions = [
        "Write a function that takes a list of numbers and returns their sum",
        "Create a function that checks if a string is a palindrome",
        "Write a function to find the factorial of a number",
        "Create a function that counts the frequency of each element in a list",
        "Write a function to find the nth Fibonacci number"
    ]
    
    # Store generated solutions
    generated_solutions = []
    
    # Generate solution for each question
    for question in test_questions:
        solution_data = generate_code_solution(model, tokenizer, question)
        generated_solutions.append(solution_data)
    
    # Save all solutions to JSON
    save_to_json(generated_solutions)

if __name__ == "__main__":
    main()