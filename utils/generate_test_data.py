"""
Utility script for generating test data with illustrative examples
"""
import pandas as pd
import yaml
import os
import random

def load_categories():
    """Load categories from coding scheme"""
    try:
        with open('coding_scheme.yml', 'r', encoding='utf-8') as file:
            scheme = yaml.safe_load(file)
            print("\nCategories found in YAML:")
            for key in scheme.keys():
                print(f"- {key}")
            return list(scheme.keys())
    except Exception as e:
        print(f"Error loading coding scheme: {e}")
        return []

def generate_test_entries():
    """Generate two illustrative test entries"""
    entries = [
        {
            'title': "Digitale Medien im Mathematikunterricht",
            'description': """In diesem Workshop lernen Lehrkräfte den Einsatz digitaler 
            Werkzeuge für den Mathematikunterricht. Schwerpunkte sind: GeoGebra für 
            geometrische Konstruktionen, Tabellenkalkulationen für Statistik, und 
            Online-Lernplattformen für individualisiertes Üben. Teilnehmende entwickeln 
            eigene digitale Unterrichtsmaterialien."""
        },
        {
            'title': "Klassenführung und Konfliktmanagement",
            'description': """Dieser Grundlagenkurs vermittelt traditionelle Methoden 
            der Klassenführung. Themen sind: Regeln aufstellen und durchsetzen, 
            Störungen vorbeugen, konstruktiver Umgang mit Konflikten, Gestaltung eines 
            positiven Lernklimas. Praxisnahe Übungen und Fallbeispiele."""
        }
    ]
    
    df = pd.DataFrame(entries)
    df.to_csv('teacher_training_data.csv', index=False)
    print("Generated 2 test entries:")
    print("\nEntry 1: Digital example")
    print(f"Title: {entries[0]['title']}")
    print(f"Description: {entries[0]['description'][:100]}...")
    print("\nEntry 2: Traditional example")
    print(f"Title: {entries[1]['title']}")
    print(f"Description: {entries[1]['description'][:100]}...")

def get_code_probability(category: str, is_digital: bool) -> float:
    """
    Get probability of code being 1 based on category and content type
    """
    # Digital competence categories (higher chance of 1 for digital content)
    if category.startswith(('2.2.', '2.4.')):
        return 0.9 if is_digital else 0.1
    
    # Teaching methodology categories (can be either)
    elif category.startswith('2.3.'):
        return 0.6 if is_digital else 0.4
    
    # General categories (random)
    else:
        return 0.5

def generate_human_codes():
    """Generate human codes for test entries with realistic variation"""
    try:
        # Read training data with both title and description
        df = pd.read_csv('teacher_training_data.csv')
        categories = load_categories()  # Get categories from YAML
        
        # Initialize DataFrame with title only (description not needed in human codes)
        codes_df = pd.DataFrame({'title': df['title']})
        
        # Generate codes for each category
        for category in categories:
            codes = []
            for _, row in df.iterrows():
                is_digital = 'digital' in row['title'].lower()
                prob = get_code_probability(category, is_digital)
                code = "1" if random.random() < prob else "0"
                codes.append(code)
            
            column_name = f"human_code_{category}"
            codes_df[column_name] = codes
        
        # Save to Excel with formatting
        with pd.ExcelWriter('human_codes.xlsx', engine='openpyxl') as writer:
            codes_df.to_excel(writer, index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Sheet1']
            for idx, col in enumerate(codes_df.columns):
                max_length = max(
                    codes_df[col].astype(str).apply(len).max(),
                    len(col)
                )
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)
        
        print(f"\nGenerated human codes for {len(categories)} categories")
        print("\nCodes for both entries:")
        print(codes_df.to_string())
        
    except Exception as e:
        print(f"Error generating human codes: {e}")

if __name__ == "__main__":
    print("Generating illustrative test dataset...")
    generate_test_entries()
    generate_human_codes() 