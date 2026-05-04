import pandas as pd
import time
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=groq_api_key,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct", 
    temperature=0.3 
)

system_prompt = """
You are an expert programming instructor. Your task is to generate a 'Socratic Hint' for a student.
You will be provided with the student's 'Buggy Code' and the 'Fixed Code'.

CRITICAL RULES:
1. DO NOT provide the Fixed Code in your response.
2. DO NOT point out the exact line or character to change directly.
3. Write a short, encouraging conceptual hint or a guiding question that helps the student realize the difference between their code and the correct approach.
4. Keep it concise (under 3 sentences).
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "Buggy Code:\n{buggy_code}\n\nFixed Code:\n{fixed_code}")
])

chain = prompt_template | llm

def generate_synthetic_dataset_robust(input_csv_path, output_csv_path):
    print("Loading dataset...")
    df = pd.read_csv(input_csv_path)
    total_rows = len(df)
    
    start_index = 0
    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
        start_index = len(existing_df)
        print(f"Found existing file with {start_index} records.")
        print(f"Resuming from index {start_index}...")
    else:
        print("No existing output file found. Starting from scratch...")
        pd.DataFrame(columns=["instruction", "output"]).to_csv(output_csv_path, index=False)

    if start_index >= total_rows:
        print("🎉 Dataset generation is already 100% complete!")
        return

    for index in range(start_index, total_rows):
        row = df.iloc[index]
        buggy_code = row['buggy_code']
        fixed_code = row['fixed_code']
        
        try:
            response = chain.invoke({
                "buggy_code": buggy_code,
                "fixed_code": fixed_code
            })
            hint = response.content.strip()
            
            new_data = pd.DataFrame([{
                "instruction": f"Analyze this buggy code and provide a Socratic hint:\n{buggy_code}",
                "output": hint
            }])
            
            new_data.to_csv(output_csv_path, mode='a', header=False, index=False, encoding='utf-8')
            
            print(f"[{index + 1}/{total_rows}] Hint saved successfully.")
            
            time.sleep(3) 
            
        except Exception as e:
            error_message = str(e)
            print(f"[{index + 1}/{total_rows}] ⚠️ Error: {error_message}")
            
            if "429" in error_message or "rate limit" in error_message.lower():
                print("🛑 Hit Rate Limit/Daily Limit! Stopping the script cleanly.")
                print("Please try again tomorrow or use a new API Key.")
                break
            else:
                time.sleep(5)

if __name__ == "__main__":
    input_file_name = 'data/unlabeled_data.csv' 
    output_file_name = 'data/labeled_data.csv'
    
    generate_synthetic_dataset_robust(input_file_name, output_file_name)
