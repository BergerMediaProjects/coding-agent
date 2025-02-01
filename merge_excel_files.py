import pandas as pd
import os

def merge_excel_files(file_prefix, folder_path="old_data"):
    """
    Merge Excel files with the same prefix in the specified folder
    """
    # List all files with the given prefix
    files = [f for f in os.listdir(folder_path) if f.startswith(file_prefix) and f.endswith('.xlsx')]
    
    # Read and concatenate all files
    dfs = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path)
        dfs.append(df)
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df

def main():
    # Merge data files
    print("Merging data files...")
    data_merged = merge_excel_files('data_')
    
    # Merge k files
    print("Merging k files...")
    k_merged = merge_excel_files('k_')
    
    # Remove duplicates from data_merged based on token
    print("Removing duplicates from data files...")
    data_merged_no_dupes = data_merged.drop_duplicates(subset=['token'], keep='first')
    print(f"Removed {len(data_merged) - len(data_merged_no_dupes)} duplicate entries from data files")
    
    # Remove duplicates from k_merged based on Lehrgangsnummer
    print("Removing duplicates from k files...")
    k_merged_no_dupes = k_merged.drop_duplicates(subset=['1.5.2 Lehrgangsnummer'], keep='first')
    print(f"Removed {len(k_merged) - len(k_merged_no_dupes)} duplicate entries from k files")
    
    # Save individual merged files
    print("Saving intermediate merged files...")
    data_merged_no_dupes.to_excel('data.xlsx', index=False)
    k_merged_no_dupes.to_excel('k.xlsx', index=False)
    
    # Merge data and k files on specified columns
    print("Performing final merge...")
    final_merged = pd.merge(
        data_merged_no_dupes,
        k_merged_no_dupes,
        left_on='token',
        right_on='1.5.2 Lehrgangsnummer',
        how='inner'  # Change to 'left', 'right', or 'outer' if needed
    )
    
    # Save final merged file
    final_merged.to_excel('merged_final.xlsx', index=False)
    
    print("\nMerging completed successfully!")
    print(f"Number of rows in data (after removing duplicates): {len(data_merged_no_dupes)}")
    print(f"Number of rows in k (after removing duplicates): {len(k_merged_no_dupes)}")
    print(f"Number of rows in final merged file: {len(final_merged)}")

if __name__ == "__main__":
    main() 