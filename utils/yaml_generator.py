import os
from docx import Document
import yaml
import re

class YAMLGenerator:
    def __init__(self):
        # Get the project root directory (parent of utils)
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def generate_key_name(self, display_name: str) -> str:
        """
        Generate a valid Python identifier for use as dictionary key.
        Example:
        - "2.0a Digitale Medien" -> "Digitale_Medien"
        - "2.0b Digitale Medien als Lernziel" -> "Digitale_Medien_als_Lernziel"
        - "2.1.1 Organisatorische Kommunikation" -> "Organisatorische_Kommunikation"
        """
        # Remove the category number and any letter suffix
        name = re.sub(r'^\d+(\.\d+)*[a-z]?\s+', '', display_name)
        
        # Replace spaces and special characters with underscores
        key = re.sub(r'[^a-zA-Z0-9]+', '_', name)
        
        # Remove leading/trailing underscores and return
        return key.strip('_')

    def parse_condition(self, values: str) -> dict:
        """Parse condition from values string"""
        # Don't treat 2.0.2 as a derived category - check for the specific pattern
        if "nur, wenn auch 2.0.4 =1" in values.lower():
            return None
        
        # Additional check for the specific 2.0.2 pattern
        if "nur, wenn auch" in values.lower() and "2.0.4" in values.lower():
            return None
        
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

    def standardize_values(self, values: str) -> str:
        """Standardize the values format"""
        values = values.strip()
        
        # Handle conditional values
        if "wenn" in values.lower():
            return values
            
        # Handle special cases
        if "ja" in values.lower() and "nein" in values.lower():
            return "Ja (1), Nein (0)"
        elif "offen" in values.lower():
            return "offen"
            
        # Handle values with newlines
        if "\n" in values:
            # Replace newlines with semicolons and clean up
            parts = [part.strip() for part in values.split("\n")]
            # Remove empty parts and join with semicolon
            return "; ".join(part for part in parts if part)
            
        # Handle values with semicolons
        if ";" in values:
            # Split by semicolon and clean each part
            parts = []
            for part in values.split(";"):
                part = part.strip()
                if part and part not in parts:  # Avoid duplicates
                    parts.append(part)
            # Join with semicolon
            return "; ".join(parts)
            
        # Handle values with commas
        if "," in values:
            # Split by comma and clean each part
            parts = []
            for part in values.split(","):
                part = part.strip()
                if part and part not in parts:  # Avoid duplicates
                    parts.append(part)
            # Join with comma
            return ", ".join(parts)
            
        # Return cleaned value
        return values.strip()

    def generate_yaml_from_docx(self, input_file: str, output_file: str) -> bool:
        """Generate YAML from DOCX file"""
        try:
            # Load DOCX
            doc = Document(input_file)
            table = doc.tables[0]
            categories = {}
            
            # Process table
            for row_idx, row in enumerate(table.rows[1:], start=2):  # Skip header row, keep track of row number
                cells = row.cells
                display_name = cells[0].text.strip()
                values = cells[1].text.strip()
                criteria = cells[2].text.strip()
                examples = cells[3].text.strip()

                # Skip empty rows
                if not display_name and not values and not criteria and not examples:
                    print(f"Skipping empty row {row_idx}")
                    continue

                # Print detailed debug info for each row
                print(f"\nProcessing row {row_idx}:")
                print(f"Raw display_name: '{display_name}'")
                print(f"Raw values: '{values}'")
                print(f"Raw criteria: '{criteria}'")
                print(f"Raw examples: '{examples}'")

                # Generate the category key name
                key_name = self.generate_key_name(display_name)
                if not key_name:
                    print(f"Warning: Could not generate key name for {display_name}")
                    continue

                # Handle derived categories
                condition = self.parse_condition(values)
                if condition:
                    key_name = "_DERIVED_" + key_name
                    print(f"Found derived category: {key_name}")
                    print(f"With condition: {condition}")

                # Process examples into a proper list
                example_list = []
                if examples:
                    # Split by newlines and filter out empty lines
                    example_list = [ex.strip() for ex in examples.split('\n') if ex.strip()]

                # Clean up criteria
                if not criteria:
                    criteria = '""'
                else:
                    # Replace newlines with spaces and clean up
                    criteria = " ".join(criteria.split())

                # Clean up values
                values = values.replace(";;", ";")  # Remove double semicolons
                values = values.replace("; ;", ";")  # Remove semicolons with spaces
                values = values.strip(";")  # Remove leading/trailing semicolons

                # Store in dict using the generated key name
                if key_name in categories:
                    print(f"WARNING: Duplicate category found: {key_name}")
                    print(f"Previous values: {categories[key_name]}")
                    print(f"New values: {values}")
                    # Make the key name unique by appending a number
                    counter = 1
                    while f"{key_name}_{counter}" in categories:
                        counter += 1
                    key_name = f"{key_name}_{counter}"
                
                categories[key_name] = {
                    "display_name": display_name,  # Keep the original display name with numbers
                    "criteria": criteria,
                    "examples": example_list,
                    "values": self.standardize_values(values)
                }
                if condition:
                    categories[key_name]["condition"] = condition

            # Create the final YAML structure
            yaml_data = {
                "coding_scheme": {
                    "version": "1.0",
                    "categories": categories
                }
            }

            # Print summary of all categories
            print("\nAll categories found:")
            for cat in sorted(categories.keys()):
                print(f"- {cat}")
            print(f"\nTotal categories: {len(categories)}")

            # Save YAML with proper formatting
            with open(output_file, "w", encoding="utf-8") as yaml_file:
                yaml.dump(yaml_data, yaml_file, 
                         allow_unicode=True, 
                         default_flow_style=False,
                         sort_keys=False,
                         indent=2,
                         width=1000,  # Prevent line wrapping
                         default_style='"')  # Use double quotes for strings
                
            print(f"\nYAML file generated successfully: {output_file}")
            return True
            
        except Exception as e:
            print(f"Error generating YAML: {str(e)}")
            return False

if __name__ == "__main__":
    generator = YAMLGenerator()
    input_file = os.path.join(generator.root_dir, "data", "DOC_coding_scheme", "doc_cs.docx")
    output_file = os.path.join(generator.root_dir, "data", "DOC_coding_scheme", "coding_scheme_imported.yml")
    
    print(f"Processing DOCX file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    
    success = generator.generate_yaml_from_docx(input_file, output_file)
    if success:
        print("YAML generation completed successfully")
    else:
        print("YAML generation failed")
