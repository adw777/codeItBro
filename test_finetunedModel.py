import torch
from unsloth import FastLanguageModel
from transformers import GenerationConfig

def load_model(model_path: str, max_length: int = 4096):
    """Load the finetuned model and tokenizer."""
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_length,
        dtype=None,
        load_in_4bit=True
    )
    
    model = FastLanguageModel.for_inference(model)

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
    """Generate a response for the given prompt."""
    # Prepare input with chat template
    inputs = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt}
    ], tokenize=False)
    
    # Tokenize and generate
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids.cuda()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
        )
    
    # Decode and clean response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    response = response.split("assistant")[-1].strip()
    return response

def main():
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load model
    model_path = "finetuned_cot_model"  # Path to your finetuned model
    model, tokenizer, generation_config = load_model(model_path)
    print("Model loaded successfully! You can start chatting.")
    print('Type "quit" to exit')
    print("-" * 50)
    
    while True:
        # Get user input
        prompt = input("\nYour prompt: ").strip()
        
        # Check for quit command
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not prompt:
            continue
        
        try:
            # Generate and print response
            print("\nGenerating response...\n")
            response = generate_response(model, tokenizer, generation_config, prompt)
            print("Response:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    main()