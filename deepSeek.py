# from bespokelabs import curator
# from pydantic import BaseModel, Field
# import json
# import re

# class NDAContent(BaseModel):
#     """Model for NDA content including thinking process and agreement text"""
#     thinking_process: str = Field(description="The model's thinking process")
#     agreement_content: str = Field(description="The actual NDA content")

# class NDAGenerator(curator.LLM):
#     """LLM class for generating NDAs with thinking process"""
#     response_format = NDAContent
    
#     def prompt(self, input: dict) -> str:
#         return """Create a legal Non-Disclosure Agreement (NDA) between an employee and a company.
#         Show your thinking process in a <think> tag, followed by the actual agreement content.
#         The thinking process should explain your reasoning and considerations.
#         The agreement should be properly formatted with clear sections and clauses.
#         """
    
#     def parse(self, input: dict, response: str) -> dict:
#         try:
#             # Extract thinking process using regex to handle potential formatting variations
#             think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
#             if not think_match:
#                 raise ValueError("Missing thinking section")
            
#             thinking = think_match.group(1).strip()
            
#             # Get content (everything after </think>)
#             content = response.split('</think>')[-1].strip()
            
#             # Create NDAContent instance
#             nda_content = NDAContent(
#                 thinking_process=thinking,
#                 agreement_content=content
#             )
            
#             # Return in the format expected by the curator framework
#             return {
#                 "thinking_process": nda_content.thinking_process,
#                 "agreement_content": nda_content.agreement_content
#             }
            
#         except Exception as e:
#             print(f"Error parsing response: {e}")
#             # Return a minimal valid response
#             return {
#                 "thinking_process": "Error processing thinking",
#                 "agreement_content": "Error processing content"
#             }

# def main():
#     # Initialize the generator
#     generator = NDAGenerator(
#         model_name="ollama/deepseek-r1:14b",
#         backend_params={"base_url": "http://localhost:11434"}
#     )
    
#     # Create dataset with one example
#     dataset = [{"input": ""}]
    
#     # Generate NDA
#     result = generator(dataset)
    
#     # Convert to pandas and display
#     df = result.to_pandas()
#     print("\nGenerated NDA with Thinking Process:")
#     print(df)
    
#     return df

# if __name__ == "__main__":
#     main()

from bespokelabs import curator
from pydantic import BaseModel, Field
import json
import re
import pandas as pd

class NDAContent(BaseModel):
    """Model for NDA content including thinking process and agreement text"""
    thinking_process: str = Field(description="The model's thinking process")
    agreement_content: str = Field(description="The actual NDA content")

class NDAGenerator(curator.LLM):
    """LLM class for generating NDAs with thinking process"""
    def prompt(self, input: dict) -> str:
        return """Create a legal Non-Disclosure Agreement (NDA) between an employee and a company.
        Show your thinking process in a <think> tag, followed by the actual agreement content.
        The thinking process should explain your reasoning and considerations.
        The agreement should be properly formatted with clear sections and clauses.
        """
    
    def parse(self, input: dict, response: str) -> dict:
        try:
            # Extract thinking process using regex to handle potential formatting variations
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            if not think_match:
                raise ValueError("Missing thinking section")
            
            thinking = think_match.group(1).strip()
            
            # Get content (everything after </think>)
            content = response.split('</think>')[-1].strip()
            
            return {
                "thinking_process": thinking,
                "agreement_content": content
            }
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            return {
                "thinking_process": "Error processing thinking",
                "agreement_content": "Error processing content"
            }

def main():
    # Initialize the generator
    generator = NDAGenerator(
        model_name="ollama/deepseek-r1:14b",  # Updated model name
        backend_params={"base_url": "http://localhost:11434"}
    )
    
    # Create dataset with one example
    dataset = [{"input": ""}]
    
    # Generate NDA
    raw_results = generator(dataset)
    
    # Convert results to DataFrame
    results_list = []
    for result in raw_results:
        parsed = generator.parse({}, result.completion)
        results_list.append(parsed)
    
    df = pd.DataFrame(results_list)
    print("\nGenerated NDA with Thinking Process:")
    print(df)
    
    return df

if __name__ == "__main__":
    main()