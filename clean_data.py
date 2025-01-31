import json

def clean_json_data(input_file='codingQues.json', output_file='cleaned_codingQues.json'):
    try:
        # Read the original JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Transform the data
        cleaned_data = []
        for item in data:
            # Extract question and topic
            question = item['input']['question']
            topic = item['input']['topic']
            
            # Combine them into a single string
            combined_text = f"{question} [Topic: {topic}]"
            
            # Add to cleaned data
            cleaned_data.append({
                "input": combined_text
            })

        # Write the cleaned data to a new JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        print(f"Successfully cleaned the data and saved to {output_file}")
        print(f"Processed {len(cleaned_data)} questions")
        
        # Print a sample of the cleaned data
        print("\nSample of cleaned data:")
        for item in cleaned_data[:2]:
            print(item)

    except FileNotFoundError:
        print(f"Error: Could not find the input file {input_file}")
    except json.JSONDecodeError:
        print(f"Error: The input file {input_file} is not valid JSON")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    clean_json_data()