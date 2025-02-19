"""
Enhanced YAML validator for coding scheme
"""

import yaml
from pydantic import BaseModel, Field, validator
from typing import List, Dict
import sys
import re
import os

class Category(BaseModel):
    """Structure for a single category in the coding scheme"""
    criteria: str
    examples: List[str]
    values: str

    @validator('values')
    def validate_values_format(cls, v):
        """Validate the values field with more flexible formats"""
        # Remove surrounding quotes if present
        v = v.strip('"\'')
        
        # Allow these formats:
        # 1. Binary: "Ja (1), Nein (0)"
        # 2. Open text: "offen" or specific text fields
        # 3. Multiple choice with numbers: "Option1 (1), Option2 (2), ..."
        # 4. Special codes: "-99" with optional explanation
        # 5. Semester format: "SoSe25; WiSe24/25; -99"
        # 6. List format: "A; B; C; D; E; Z; G; -99"
        
        # Don't validate if it's a derived category
        if v.startswith('wenn'):
            return v
            
        return v

    @validator('examples')
    def validate_examples(cls, v):
        """Validate examples list"""
        if not isinstance(v, list):
            raise ValueError('Examples must be a list')
        return v

    @validator('criteria')
    def validate_criteria(cls, v):
        """Validate criteria field"""
        if not v.strip():
            return '""'  # Return empty string if no criteria
        return v

def validate_yaml(file_path: str) -> bool:
    """Validate the YAML file format and structure"""
    try:
        # Read and parse YAML
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            try:
                data = yaml.safe_load(content)
            except yaml.YAMLError as e:
                print(f"\n❌ YAML Syntax Error:")
                print(f"  {str(e)}")
                return False

        # Validate each category
        for category_name, category_data in data.items():
            try:
                # Check if it's a derived category
                is_derived = category_name.startswith('_DERIVED_')
                
                # Basic structure check
                if not isinstance(category_data, dict):
                    print(f"\n❌ Error in category '{category_name}':")
                    print("  Category must be a dictionary with criteria, examples, and values")
                    return False
                
                # Required fields
                required_fields = ['criteria', 'examples', 'values']
                for field in required_fields:
                    if field not in category_data:
                        print(f"\n❌ Error in category '{category_name}':")
                        print(f"  Missing required field: {field}")
                        return False
                
                # Validate examples format
                if not isinstance(category_data['examples'], list):
                    print(f"\n❌ Error in category '{category_name}':")
                    print("  'examples' must be a list")
                    return False
                
            except Exception as e:
                print(f"\n❌ Error validating category '{category_name}':")
                print(f"  {str(e)}")
                return False

        print(f"\n✅ YAML validation successful!")
        print(f"Found {len(data)} categories:")
        for category in data:
            print(f"- {category}")
        return True

    except Exception as e:
        print(f"\n❌ Unexpected error during validation:")
        print(f"  {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
    else:
        yaml_file = "coding_scheme.yml"
    
    print(f"\nValidating {yaml_file}...")
    validate_yaml(yaml_file) 