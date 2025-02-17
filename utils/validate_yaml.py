"""
Enhanced YAML validator with stricter formatting rules
"""

import yaml
from pydantic import BaseModel, Field, validator
from typing import List, Dict
import sys
import re

class Category(BaseModel):
    """Structure for a single category in the coding scheme"""
    criteria: str
    examples: List[str]
    values: str

    @validator('values')
    def validate_values_format(cls, v):
        if not re.match(r'^Ja \(1\)(,| und) Nein \(0\)$', v):
            raise ValueError('Values must be in format: "Ja (1), Nein (0)"')
        return v

    @validator('examples')
    def validate_examples(cls, v):
        if not v:  # Check if empty
            raise ValueError('Must have at least one example')
        return v

class CodingScheme(BaseModel):
    """Complete coding scheme structure"""
    categories: Dict[str, Category]

def validate_yaml(file_path: str) -> bool:
    """
    Validate the YAML file format and structure
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        bool: True if valid, False if invalid
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            
        # Check formatting
        issues = []
        for category_name, content in data.items():
            # Check category name format
            if any(char in category_name for char in [':', '{', '}', '[', ']']):
                issues.append(f"Invalid characters in category name: {category_name}")
            
            # Check indentation and structure
            if not isinstance(content, dict):
                issues.append(f"Invalid structure for category: {category_name}")
            elif not all(key in content for key in ['criteria', 'examples', 'values']):
                issues.append(f"Missing required fields in category: {category_name}")

        if issues:
            print("\n❌ Formatting issues found:")
            for issue in issues:
                print(f"- {issue}")
            return False

        # Validate structure
        scheme = {name: Category(**cat) for name, cat in data.items()}
        
        print(f"\n✅ YAML is valid!")
        print(f"\nFound {len(scheme)} categories:")
        for category in scheme:
            print(f"- {category}")
            
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
    else:
        yaml_file = "coding_scheme.yml"
    
    print(f"Validating {yaml_file}...")
    validate_yaml(yaml_file) 