"""
Enhanced YAML validator with stricter formatting rules
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
        # Remove surrounding quotes if present
        v = v.strip('"\'')
        
        # Allow more flexible value formats:
        # 1. Standard binary: "Ja (1), Nein (0)"
        # 2. Multiple choice: e.g., "Seminar (1), Coaching (2), Event (3), weitere (offen)"
        # 3. Open text: e.g., "offen" or "Text"
        # 4. Special codes: e.g., "-99 wenn nicht angegeben"
        
        valid_formats = [
            # Binary format
            r'^Ja \(1\)(,| und) Nein \(0\)$',
            # Multiple choice with numbers
            r'^([^()]+\s*\(\d+\)\s*,\s*)*[^()]+\s*\(\d+\)$',
            # Open text indicator
            r'^(offen|Text)$',
            # Special codes
            r'^-?\d+.*$'
        ]
        
        if not any(re.match(pattern, v) for pattern in valid_formats):
            raise ValueError(
                'Values must be in one of these formats:\n'
                '- "Ja (1), Nein (0)"\n'
                '- "Option1 (1), Option2 (2), ..."\n'
                '- "offen" or "Text"\n'
                '- Special codes like "-99 wenn..."'
            )
        return v

    @validator('examples')
    def validate_examples(cls, v):
        if not v:  # Check if empty
            raise ValueError('Must have at least one example')
        return v

class CodingScheme(BaseModel):
    """Complete coding scheme structure"""
    categories: Dict[str, Category]

def get_line_number(file_content: str, search_text: str) -> int:
    """Get line number for a piece of text in the file"""
    lines = file_content.split('\n')
    for i, line in enumerate(lines, 1):
        if search_text in line:
            return i
    return -1

def validate_yaml(file_path: str) -> bool:
    """Validate the YAML file format and structure with line numbers"""
    try:
        if not os.path.exists(file_path):
            print(f"\n❌ Error: File not found: {file_path}")
            return False

        # Read file content first to get line numbers
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            try:
                data = yaml.safe_load(file_content)
            except yaml.YAMLError as e:
                if hasattr(e, 'problem_mark'):
                    line = e.problem_mark.line + 1
                    print(f"\n❌ YAML Syntax Error at line {line}:")
                    print(f"  {str(e)}")
                else:
                    print(f"\n❌ YAML Syntax Error: {str(e)}")
                return False
            
        # Check formatting with line numbers
        issues = []
        for category_name, content in data.items():
            line_num = get_line_number(file_content, category_name)
            
            # Check category name format
            if any(char in category_name for char in [':', '{', '}', '[', ']']):
                issues.append(f"Line {line_num}: Invalid characters in category name: {category_name}")
            
            # Check structure
            if not isinstance(content, dict):
                issues.append(f"Line {line_num}: Invalid structure for category: {category_name}")
            elif not all(key in content for key in ['criteria', 'examples', 'values']):
                missing = [k for k in ['criteria', 'examples', 'values'] if k not in content]
                issues.append(f"Line {line_num}: Missing fields {missing} in category: {category_name}")

        if issues:
            print("\n❌ Formatting issues found:")
            for issue in issues:
                print(f"- {issue}")
            return False

        # Validate structure with line numbers
        try:
            scheme = {name: Category(**cat) for name, cat in data.items()}
        except Exception as e:
            print("\n❌ Validation Error:")
            print(f"- {str(e)}")
            # Try to identify which category caused the error
            for name, cat in data.items():
                try:
                    Category(**cat)
                except Exception as cat_error:
                    line_num = get_line_number(file_content, name)
                    print(f"- Error at line {line_num} in category '{name}': {str(cat_error)}")
            return False
        
        print(f"\n✅ YAML is valid!")
        print(f"\nFound {len(scheme)} categories:")
        for category in scheme:
            print(f"- {category}")
            
        return True
        
    except Exception as e:
        print(f"\n❌ Unexpected Error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
    else:
        yaml_file = "coding_scheme.yml"
    
    print(f"Validating {yaml_file}...")
    validate_yaml(yaml_file) 