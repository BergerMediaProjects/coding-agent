from openai import OpenAI
import pandas as pd
import json
from sklearn.metrics import cohen_kappa_score
import os

# Initialize OpenAI client (Replace with your actual key or use environment variables)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"))

# Load dataset (Replace with your dataset path)
def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

# Load structured coding scheme from JSON
def load_coding_scheme(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)

# Function to construct and send the prompt to ChatGPT
def generate_ai_code(description: str, coding_scheme: dict) -> tuple[str, float]:
    prompt = f"""
    You are an AI model trained to classify teacher training descriptions into categories. 
    Use the following coding scheme: {json.dumps(coding_scheme, indent=2)}
    
    Instruction:
    - Read the training description carefully.
    - Assign the most appropriate category from the coding scheme.
    - Return only the category label and a confidence score (0-100).
    
    Description: "{description}"
    """
    
    response = client.chat.completions.create(
        model="gpt-4",  
        messages=[
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    ai_output = response.choices[0].message.content
    
    # Extract category and confidence (Assumes structured output from GPT)
    try:
        result = json.loads(ai_output)
        return result.get("category", "Unknown"), float(result.get("confidence", 0))
    except json.JSONDecodeError:
        return "Unknown", 0.0

# Process dataset and apply AI coding
def process_data(dataset: pd.DataFrame, coding_scheme: dict) -> pd.DataFrame:
    ai_results = []
    
    for _, row in dataset.iterrows():
        description = row["description"]  # Adjust column name if necessary
        ai_category, confidence = generate_ai_code(description, coding_scheme)
        
        ai_results.append({
            "description": description,
            "human_code": row.get("human_code", "Unknown"),
            "ai_code": ai_category,
            "confidence": confidence
        })
    
    return pd.DataFrame(ai_results)

# Compare AI coding to human coding using Cohen's Kappa
def evaluate_results(df: pd.DataFrame):
    human_labels = df["human_code"].fillna("Unknown")
    ai_labels = df["ai_code"]
    
    kappa = cohen_kappa_score(human_labels, ai_labels)
    print(f"Cohen's Kappa Score (AI vs Human): {kappa:.2f}")

# Main execution
def main():
    dataset = load_dataset("teacher_training_data.csv")
    coding_scheme = load_coding_scheme("old_data_training/coding_scheme.json")
    
    results_df = process_data(dataset, coding_scheme)
    
    # Save AI-coded results
    results_df.to_csv("ai_coded_results.csv", index=False)
    
    # Evaluate performance
    evaluate_results(results_df)
    
    # Display results
    print("\nAI Coding Results:")
    print(results_df.to_string())

if __name__ == "__main__":
    main()
