"""
Data preparation script to convert Excel files to CSV format
"""

import pandas as pd
import os

def convert_training_data():
    """Convert merged_final.xlsx to teacher_training_data.csv"""
    try:
        # Read Excel file
        df = pd.read_excel("merged_final.xlsx")
        
        # Ensure required columns exist
        required_columns = ['title', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean data
        df['title'] = df['title'].str.strip()  # Remove whitespace
        df['description'] = df['description'].str.strip()
        
        # Remove empty entries
        df = df.dropna(subset=['title', 'description'])
        
        # Select only necessary columns
        df = df[['title', 'description']]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        # Save as CSV
        output_path = "teacher_training_data.csv"
        df.to_csv(output_path, index=False)
        print(f"Successfully converted to {output_path}")
        print(f"Total training entries: {len(df)}")
        print(f"Columns in output: {', '.join(df.columns)}")
        
    except FileNotFoundError:
        print("Error: merged_final.xlsx not found")
    except Exception as e:
        print(f"Error during training data conversion: {str(e)}")

def convert_human_codes():
    """Convert old_data_ink19_codes.xlsx to human_codes.csv"""
    try:
        # Read Excel file
        df = pd.read_excel("old_data_training/old_data_ink19_codes.xlsx")
        
        # Ensure required columns exist
        required_columns = ['title', 'human_code']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean data
        df['title'] = df['title'].str.strip()  # Remove whitespace
        
        # Select and clean necessary columns
        df = df[['title', 'human_code']]
        
        # Remove empty entries
        df = df.dropna(subset=['title', 'human_code'])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        # Ensure human_code is in correct format (0 or 1)
        df['human_code'] = df['human_code'].astype(str)
        invalid_codes = df[~df['human_code'].isin(['0', '1'])]
        if not invalid_codes.empty:
            print(f"Warning: Found {len(invalid_codes)} invalid codes. Converting to '0'")
            df.loc[~df['human_code'].isin(['0', '1']), 'human_code'] = '0'
        
        # Save as CSV
        output_path = "human_codes.csv"
        df.to_csv(output_path, index=False)
        print(f"Successfully converted to {output_path}")
        print(f"Total human codes: {len(df)}")
        print(f"Columns in output: {', '.join(df.columns)}")
        
    except FileNotFoundError:
        print("Error: old_data_training/old_data_ink19_codes.xlsx not found")
    except Exception as e:
        print(f"Error during human codes conversion: {str(e)}")

if __name__ == "__main__":
    print("Converting training data...")
    convert_training_data()
    print("\nConverting human codes...")
    convert_human_codes() 