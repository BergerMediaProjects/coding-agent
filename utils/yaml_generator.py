import os
from docx import Document
import yaml

# Get the project root directory (parent of utils)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define input/output paths
doc_folder = os.path.join(root_dir, "DOC_coding_scheme")
input_file = os.path.join(doc_folder, "doc_cs.docx")
output_file = os.path.join(doc_folder, "coding_scheme_imported.yml")

# Load DOCX file using absolute path
doc = Document(input_file)

# Extract table
table = doc.tables[0]  # Assuming the coding table is the first table

# Prepare structured data
codes = {}  # Changed from list to dict

def parse_condition(values: str) -> dict:
    """Parse condition from values string"""
    if "wenn" not in values.lower():
        return None
        
    # Extract condition
    condition = values.lower().split("wenn")[1].strip()
    
    # Parse different condition types
    if "min. eine der kategorien" in condition:
        # Format: "wenn min. eine der Kategorien 2.1.1-2.1.4 = 1"
        try:
            # Extract range
            range_part = condition.split("kategorien")[1].split("=")[0].strip()
            start, end = range_part.split("-")
            # Extract value
            value = condition.split("=")[1].strip()
            return {
                "type": "any_in_range",
                "range_start": start.strip(),
                "range_end": end.strip(),
                "value": value
            }
        except Exception as e:
            print(f"Warning: Could not parse range condition: {condition}")
            return None
    elif "=" in condition:
        # Simple equality check
        ref_category, value = condition.split("=")
        return {
            "type": "equals",
            "reference": ref_category.strip(),
            "value": value.strip()
        }
    return None

def standardize_values(values: str) -> str:
    """Standardize the values format"""
    # Clean up the string
    values = values.strip()
    base_values = values.split("wenn")[0].strip() if "wenn" in values.lower() else values
    
    # Handle different formats
    if "ja" in base_values.lower() and "nein" in base_values.lower():
        return "Ja (1), Nein (0)"
    elif "offen" in base_values.lower():
        return "offen"
    elif any(x in base_values for x in [";", ","]):
        return base_values.replace("\n", " ").strip()
    else:
        return base_values

print("\nProcessing table rows:")
for row in table.rows[1:]:  # Skip header row if present
    cells = row.cells
    category = cells[0].text.strip()
    values = cells[1].text.strip()
    criteria = cells[2].text.strip()
    examples = cells[3].text.strip()

    # Handle derived categories
    condition = parse_condition(values)
    if condition:
        category = "_DERIVED_" + category
        
    print(f"\nCategory: {category}")
    print(f"Values (raw): {values}")
    print(f"Examples (raw): {examples}")

    # Store in dict
    codes[category] = {
        "criteria": criteria,
        "examples": examples.split('\n') if examples else [''],
        "values": standardize_values(values)
    }
    if condition:
        codes[category]["condition"] = condition

print("\nGenerated YAML structure:")
print(yaml.dump(codes, allow_unicode=True, default_flow_style=False))

# Save YAML using absolute path
with open(output_file, "w", encoding="utf-8") as yaml_file:
    yaml.dump(codes, yaml_file, allow_unicode=True, default_flow_style=False)

print(f"YAML file created successfully in: {output_file}")
