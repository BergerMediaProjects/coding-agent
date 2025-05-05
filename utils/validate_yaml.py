"""
Enhanced YAML validator for coding scheme
"""

import yaml
from pydantic import BaseModel, Field, validator
from typing import List, Dict
import sys
import re
import os
import logging

# Get logger
logger = logging.getLogger('yaml_validator')

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
        
        # Don't validate if it's a derived category or conditional value
        if v.startswith('wenn'):
            return v
            
        # Check for valid formats
        valid_formats = [
            r'^Ja \(1\), Nein \(0\)$',  # Binary format
            r'^offen$',  # Open text
            r'^[^,]+ \(\d+\), [^,]+ \(\d+\)(?:, [^,]+ \(\d+\))*$',  # Multiple choice with numbers
            r'^[^,]+ \(\d+\), [^,]+ \(\d+\), -99$',  # Multiple choice with -99
            r'^SoSe\d{2}; WiSe\d{2}/\d{2}; -99$',  # Semester format
            r'^[A-Z]; [A-Z](?:; [A-Z])*; -99$',  # List format with letters
            r'^[^,]+; [^,]+(?:; [^,]+)*; -99$',  # List format with text
            r'^[^,]+(?:, [^,]+)*$',  # Simple comma-separated list
            r'^-99$'  # Just -99
        ]
        
        # Check if value matches any valid format
        if not any(re.match(pattern, v) for pattern in valid_formats):
            raise ValueError(f'Invalid values format: {v}')
            
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
                logger.error(f"YAML Syntax Error: {str(e)}")
                return False

        if not isinstance(data, dict):
            logger.error("Root element must be a dictionary")
            return False

        # Check for coding_scheme root object
        if 'coding_scheme' not in data:
            logger.error("Missing 'coding_scheme' root object")
            return False

        coding_scheme = data['coding_scheme']
        if not isinstance(coding_scheme, dict):
            logger.error("'coding_scheme' must be a dictionary")
            return False

        # Check for required fields in coding_scheme
        if 'version' not in coding_scheme:
            logger.error("Missing 'version' field in coding_scheme")
            return False

        if 'categories' not in coding_scheme:
            logger.error("Missing 'categories' field in coding_scheme")
            return False

        if not isinstance(coding_scheme['categories'], dict):
            logger.error("'categories' must be a dictionary")
            return False

        # Check for duplicate category names
        category_names = {}  # dict to store category names and their keys
        for category_key, category_data in coding_scheme['categories'].items():
            if not isinstance(category_data, dict):
                logger.error(f"Error in category '{category_key}': Category must be a dictionary")
                return False
            
            # Get the display name or use the key if display_name is not present
            display_name = category_data.get('display_name', category_key)
            
            # Extract the actual category name by removing only the number prefix
            # Match the pattern: digits and dots at the start, optionally followed by a letter and space
            match = re.match(r'^\d+(?:\.\d+)*(?:\s*[a-z])?\s+(.+)$', display_name, re.IGNORECASE)
            if match:
                category_name = match.group(1)  # Get the actual name part
            else:
                category_name = display_name
            
            # Check if we've seen this name before
            if category_name in category_names:
                logger.error(f"Duplicate category name found: '{category_name}' in categories '{category_names[category_name]}' and '{display_name}'")
                return False
            category_names[category_name] = display_name
            
            # Required fields
            required_fields = ['criteria', 'examples', 'values']
            for field in required_fields:
                if field not in category_data:
                    logger.error(f"Error in category '{category_key}': Missing required field: {field}")
                    return False
            
            # Validate examples format
            if not isinstance(category_data['examples'], list):
                logger.error(f"Error in category '{category_key}': 'examples' must be a list")
                return False
            
            # Validate values format
            if not isinstance(category_data['values'], str):
                logger.error(f"Error in category '{category_key}': 'values' must be a string")
                return False
            
            # Validate criteria format
            if not isinstance(category_data['criteria'], str):
                logger.error(f"Error in category '{category_key}': 'criteria' must be a string")
                return False

        logger.info(f"YAML validation successful! Found {len(coding_scheme['categories'])} categories")
        return True

    except Exception as e:
        logger.error(f"Unexpected error during validation: {str(e)}")
        return False

if __name__ == "__main__":
    # Setup basic logging for command line usage
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
    else:
        yaml_file = "coding_scheme.yml"
    
    logger.info(f"Validating {yaml_file}...")
    validate_yaml(yaml_file) 