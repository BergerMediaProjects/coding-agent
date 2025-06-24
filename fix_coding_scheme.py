import yaml
import re

def clean_name(name):
    # Remove version numbers and special characters
    name = re.sub(r'^\d+\.\d+(\.\d+)?[a-z]?\s*', '', name)
    # Remove any remaining numbers and special characters
    name = re.sub(r'[^a-zA-ZäöüÄÖÜß\s]', '', name)
    # Convert to title case and remove extra spaces
    name = ' '.join(name.split())
    return name

def add_simplified_names(yaml_file):
    # Read the YAML file
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # Add simplified_name to each category
    for category, details in data['coding_scheme']['categories'].items():
        if 'display_name' in details:
            details['simplified_name'] = clean_name(details['display_name'])
    
    # Write back to the file
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

if __name__ == '__main__':
    yaml_file = 'data/DOC_coding_scheme/coding_scheme_imported.yml'
    add_simplified_names(yaml_file) 