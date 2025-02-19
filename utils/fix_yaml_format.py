"""
Script to automatically fix formatting issues in coding_scheme.yml
"""

import yaml
import re
from typing import Dict, Any
import os

def simplify_category_name(name: str) -> str:
    """
    Remove numbering pattern from category names
    Examples:
    - "2.0 a Vorkommen Medienkompetenz" -> "Vorkommen Medienkompetenz"
    - "2.1.1 Zielkompetenz" -> "Zielkompetenz"
    - ".1 Genauer Name" -> "Genauer Name"
    - "3 DigComp" -> "DigComp"
    """
    # Remove various numbering patterns
    patterns = [
        r'^\d+\.\d+\.?\d*\s*[a-z]?\s*',  # Matches "2.0 a", "2.1.1", etc.
        r'^\.\d+\s*',                     # Matches ".1", ".2", etc.
        r'^\d+\s+'                        # Matches "3 ", "4 ", etc.
    ]
    
    result = name
    for pattern in patterns:
        result = re.sub(pattern, '', result)
    
    return result.strip()

def fix_criteria(criteria: str) -> str:
    """Fix formatting of criteria text"""
    # Remove 'Kriterium:' prefix
    criteria = re.sub(r'^Kriterium:\s*', '', criteria)
    # Fix quotes and line breaks
    criteria = criteria.replace('"', "'").strip()
    return criteria

def fix_examples(examples: list) -> list:
    """Fix formatting of examples"""
    fixed = []
    for example in examples:
        # Handle dict-type examples
        if isinstance(example, dict):
            # Convert dict to string, e.g. {"Begriffe": "Fähigkeit"} -> "Begriffe: Fähigkeit"
            example = "; ".join(f"{k}: {v}" for k, v in example.items())
        
        # Remove bullet points and fix quotes
        example = re.sub(r'^[·•]\s*', '', str(example))
        example = example.strip().replace('"', "'")
        
        # Add quotes if not present
        if not (example.startswith('"') and example.endswith('"')):
            example = f'"{example}"'
        fixed.append(example)
    return fixed

def fix_values(values: str) -> str:
    """Fix formatting of values while preserving content"""
    if not values:
        return '""'  # Return empty string if no values
        
    # Convert to string if not already
    values = str(values)
    
    # Remove extra quotes and whitespace
    values = values.strip().strip('"\'')
    
    # Add consistent double quotes
    if not values.startswith('"'):
        values = f'"{values}"'
        
    return values

def fix_yaml_format(input_file: str, output_file: str):
    """Fix YAML format and simplify category names"""
    try:
        # Read the original YAML
        with open(input_file, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        # Create new dict with simplified keys and fixed content
        fixed_data = {}
        for key, value in data.items():
            new_key = simplify_category_name(key)
            # Fix the content structure
            fixed_content = {
                'criteria': fix_criteria(value['criteria']),
                'examples': fix_examples(value['examples']),
                'values': fix_values(value.get('values', ''))  # Use get() to handle missing values
            }
            fixed_data[new_key] = fixed_content
            print(f"Converted category:\n  From: {key}\n  To:   {new_key}")
            print(f"  Values: {fixed_content['values']}")  # Debug print
        
        # Write the fixed YAML
        with open(output_file, 'w', encoding='utf-8') as file:
            yaml.dump(fixed_data, file, allow_unicode=True, sort_keys=False)
        
        print(f"\nFixed YAML saved to: {output_file}")
        print("New categories:")
        for category in fixed_data.keys():
            print(f"- {category}")
            
    except Exception as e:
        print(f"Error fixing YAML: {str(e)}")

if __name__ == "__main__":
    # Get the project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define input/output paths
    input_file = os.path.join(root_dir, "DOC_coding_scheme", "coding_scheme_imported.yml")
    output_file = os.path.join(root_dir, "coding_scheme.yml")
    
    fix_yaml_format(input_file, output_file) 