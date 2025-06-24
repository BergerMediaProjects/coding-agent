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
    
    # Handle conditional values
    if "wenn" in values.lower():
        return f'"{values}"'
    
    # Handle special cases
    if values.lower() == "offen":
        return '"offen"'
    
    # Handle values with double semicolons
    values = values.replace(";;", ";")
    values = values.replace("; ;", ";")
    
    # Handle values with mixed separators
    if ";" in values or "," in values:
        # Split by either semicolon or comma
        parts = []
        for part in values.replace(';', ',').split(','):
            part = part.strip()
            if part:  # Only add non-empty parts
                parts.append(part)
        # Join with consistent separator
        values = ", ".join(parts)
    
    # Handle special codes
    if "-99" in values:
        # Ensure proper spacing around -99
        values = values.replace(" -99", " -99")
        values = values.replace("-99 ", "-99 ")
    
    # Add consistent double quotes
    if not values.startswith('"'):
        values = f'"{values}"'
        
    return values

def fix_yaml_format(input_file: str, output_file: str) -> bool:
    """Fix YAML format and simplify category names"""
    try:
        # Read the original YAML
        with open(input_file, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        if not isinstance(data, dict):
            print(f"Error: Input YAML is not a dictionary")
            return False
            
        # Get the categories from the input data
        input_categories = {}
        if 'coding_scheme' in data and isinstance(data['coding_scheme'], dict):
            input_categories = data['coding_scheme'].get('categories', {})
        else:
            input_categories = data  # Assume the whole dict is categories if no coding_scheme wrapper
            
        # Create new dict with simplified keys and fixed content
        fixed_data = {}
        for key, value in input_categories.items():
            # Include all categories, including derived ones
            if not isinstance(value, dict):
                print(f"Error: Category {key} value is not a dictionary")
                return False
                
            # Keep both original and simplified names
            original_key = key
            simplified_key = simplify_category_name(key)
            
            # Fix the content structure
            try:
                fixed_content = {
                    'display_name': original_key,  # Keep original name as display_name
                    'simplified_name': simplified_key,  # Add simplified name for lookups
                    'criteria': fix_criteria(value.get('criteria', '')),
                    'examples': fix_examples(value.get('examples', [])),
                    'values': fix_values(value.get('values', ''))
                }
                
                # Add condition if it exists (for derived categories)
                if 'condition' in value:
                    fixed_content['condition'] = value['condition']
                
                # Use the original key to maintain numbering in the output
                fixed_data[original_key] = fixed_content
                print(f"Converted category:\n  From: {original_key}\n  To:   {simplified_key}")
                print(f"  Values: {fixed_content['values']}")  # Debug print
            except Exception as e:
                print(f"Error processing category {key}: {str(e)}")
                return False
        
        # Create the final structure with coding_scheme wrapper
        final_data = {
            'coding_scheme': {
                'version': '1.0',
                'categories': fixed_data
            }
        }
        
        # Write the fixed YAML
        with open(output_file, 'w', encoding='utf-8') as file:
            yaml.dump(final_data, file, 
                     allow_unicode=True, 
                     sort_keys=False,
                     default_flow_style=False,
                     indent=2,
                     width=1000)  # Increase width to prevent line wrapping
        
        print(f"\nFixed YAML saved to: {output_file}")
        print("Categories included in YAML:")
        for category in sorted(fixed_data.keys()):
            print(f"- {category}")
            
        return True
            
    except Exception as e:
        print(f"Error fixing YAML: {str(e)}")
        return False

if __name__ == "__main__":
    # Get the project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define input/output paths
    input_file = os.path.join(root_dir, "data", "DOC_coding_scheme", "coding_scheme_imported.yml")
    output_file = os.path.join(root_dir, "coding_scheme.yml")
    
    fix_yaml_format(input_file, output_file) 