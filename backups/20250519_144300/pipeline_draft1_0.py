"""
AI Training Data Classification Pipeline

Key Components:
1. Data Structures: Define the shape and validation of data throughout the pipeline
2. Management Components: Handle data loading, resource management, and results
3. AI Components: Handle GPT interaction and response validation
4. Pipeline Coordinator: Orchestrates the entire classification process

Flow:
1. Load and validate input data and coding scheme
2. For each entry:
   a. Generate category-specific prompts
   b. Get GPT classification
   c. Validate and interpret responses
3. Save results and calculate agreement metrics
"""

from openai import OpenAI
import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score
import os
import json
from typing import Dict, List, AsyncGenerator, Optional, Any
from pydantic import BaseModel, Field
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from enum import Enum
import sys
from collections import defaultdict
from docx import Document
import re

# Get the root directory path
root_dir = os.path.dirname(os.path.abspath(__file__))

# Verify API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Debug: Print environment variables
print("\n=== Environment Variables ===")
print(f"Current directory: {os.getcwd()}")
print(f"Root directory: {root_dir}")
print(f"OPENAI_API_KEY exists: {'OPENAI_API_KEY' in os.environ}")
if 'OPENAI_API_KEY' in os.environ:
    print(f"OPENAI_API_KEY length: {len(api_key)}")
    print(f"OPENAI_API_KEY first 8 chars: {api_key[:8]}...")
print("==========================\n")

# Configuration for file paths, logging, and GPT settings
CONFIG = {
    'paths': {
        'data_csv': "data/training_data.xlsx",        # Changed from CSV to Excel
        'human_codes': "data/human_codes.xlsx",              
        'coding_scheme': "data/coding_scheme.yml",           
        'prompt_template': "data/prompt.txt",                
        'output_dir': "data/results",                        
        'log_dir': "data/log",  
        'output_base': "data/results/results",  # Changed to match the expected filename pattern
        'docx_file': "data/DOC_coding_scheme/doc_cs.docx"    # Added path to DOCX file
    },
    'logging': {
        'level': 'INFO',
        'max_bytes': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5,
        'format': {
            'timestamp': '%(asctime)s',
            'level': '%(levelname)s',
            'logger': '%(name)s',
            'message': '%(message)s',
            'module': '%(module)s',
            'function': '%(funcName)s',
            'line': '%(lineno)d'
        }
    },
    'gpt': {
        'model': 'gpt-4',                              # GPT model to use
        'temperature': 0.0                             # 0.0 for most consistent results
    },
    'test_mode': {
        'enabled': True,
        'max_entries': 2,  # Process only 2 entries
        'max_categories': 2  # Process only 2 categories
    }
}

# ============================================================================
# Data Structures - Define and validate data shapes throughout the pipeline
# ============================================================================

class DataEntry(BaseModel):
    """
    Single training data entry with validation
    
    Fields:
    - title: Training title
    - description: Training description to classify
    - human_code: Optional human-assigned classification (0 or 1)
    """
    title: str
    description: str
    human_code: str = Field(
        default="0",
        pattern="^[01]$",
        description="Human-assigned code (0 or 1)"
    )

class ProcessingResult(BaseModel):
    """Model for classification results"""
    title: str
    description: str
    category: str
    ai_code: str  # Keep as string to be more flexible
    confidence: float
    reasoning: str

class ConfidenceLevel(str, Enum):
    """
    Classification confidence levels based on GPT's confidence score
    
    Thresholds:
    - LOW: 0.0 to 0.33 - Limited confidence in classification
    - MEDIUM: 0.34 to 0.66 - Moderate confidence in classification
    - HIGH: 0.67 to 1.0 - Strong confidence in classification
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ValidationResult(BaseModel):
    """Structure for validated classification results"""
    value: str  # Remove pattern validation to accept any string
    confidence: float = Field(..., ge=0.0, le=1.0, description="Raw confidence score")
    confidence_level: ConfidenceLevel = Field(..., description="Interpreted confidence level")
    reasoning: str = Field(..., description="Classification reasoning")

class CodingSchemeCategory(BaseModel):
    """Structure for a single category in the coding scheme"""
    display_name: str
    simplified_name: str
    criteria: str
    examples: List[str]
    values: str

class CodingScheme(BaseModel):
    """Structure for coding scheme"""
    version: str
    categories: Dict[str, CodingSchemeCategory]
    
    @classmethod
    def from_yaml(cls, data: Dict) -> 'CodingScheme':
        """Create CodingScheme from YAML data"""
        if not isinstance(data, dict):
            raise ValueError("YAML data must be a dictionary")
            
        if 'coding_scheme' not in data:
            raise ValueError("Missing 'coding_scheme' root object")
            
        scheme_data = data['coding_scheme']
        if not isinstance(scheme_data, dict):
            raise ValueError("coding_scheme must be a dictionary")
            
        if 'version' not in scheme_data:
            raise ValueError("coding_scheme must have 'version' field")
            
        if 'categories' not in scheme_data:
            raise ValueError("coding_scheme must have 'categories' field")
            
        if not isinstance(scheme_data['categories'], dict):
            raise ValueError("categories must be a dictionary")
            
        return cls(
            version=scheme_data['version'],
            categories=scheme_data['categories']
        )

# ============================================================================
# Management Components
# ============================================================================

class DataManager:
    """
    Manages loading and processing of training data
    
    Responsibilities:
    1. Load CSV data files
    2. Merge human codes with training data
    3. Generate validated DataEntry objects
    """
    @staticmethod
    async def load_data(path: str) -> pd.DataFrame:
        """Load and validate data file (CSV or XLSX)"""
        try:
            # Determine file type from extension
            if path.endswith('.xlsx'):
                df = pd.read_excel(path)
                print(f"\nLoaded Excel file: {path}")
                print(f"Columns found: {df.columns.tolist()}")
                return df
            else:  # default to CSV
                return pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Data file is empty: {path}")
        except Exception as e:
            raise ValueError(f"Error loading data file: {str(e)}")
    
    @staticmethod
    async def merge_datasets(main_df: pd.DataFrame, codes_df: pd.DataFrame) -> pd.DataFrame:
        """Merge training data with human codes"""
        # Merge on title
        merged = main_df.merge(codes_df, on='title', how='left')
        # Fill missing codes with "0"
        code_columns = [col for col in merged.columns if col.startswith('human_code_')]
        for col in code_columns:
            merged[col] = merged[col].fillna("0").astype(str)
        return merged
    
    @staticmethod
    async def iterate_entries(df: pd.DataFrame, category: str) -> AsyncGenerator[DataEntry, None]:
        """
        Iterate through entries with correct human code for each category
        
        Args:
            df: DataFrame with entries and human codes
            category: Current category being processed
        """
        human_code_col = f"human_code_{category}"
        for _, row in df.iterrows():
            yield DataEntry(
                title=row["title"],
                description=row["description"],
                human_code=str(row.get(human_code_col, "0"))
            )

class ResourceManager:
    """Manages loading and access to resources like coding scheme and prompts"""
    
    @staticmethod
    async def load_scheme(path: str) -> CodingScheme:
        """Load and validate coding scheme"""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
            
            # Create a mapping of simplified names to original names
            if 'coding_scheme' in data and 'categories' in data['coding_scheme']:
                categories = data['coding_scheme']['categories']
                simplified_map = {}
                for key, value in categories.items():
                    if 'simplified_name' in value:
                        simplified_map[value['simplified_name']] = key
                    
                # Create new categories dict with both original and simplified keys
                new_categories = {}
                for key, value in categories.items():
                    new_categories[key] = value  # Keep original key
                    if 'simplified_name' in value:
                        new_categories[value['simplified_name']] = value  # Add simplified key
                
                data['coding_scheme']['categories'] = new_categories
            
            return CodingScheme.from_yaml(data)
            
        except Exception as e:
            print(f"Error loading coding scheme: {str(e)}")
            raise

    @staticmethod
    async def load_template(path: str) -> str:
        """Load prompt template"""
        with open(path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    @staticmethod
    async def construct_prompt(template: str, entry: DataEntry, scheme: CodingScheme, category_key: str) -> str:
        """Construct prompt for classification"""
        try:
            category = scheme.categories.get(category_key)
            if not category:
                raise ValueError(f"Category {category_key} not found in scheme")
            
            # Use display_name if available, otherwise use the key
            display_name = category.display_name if hasattr(category, 'display_name') else category_key
            
            # Replace placeholders in template
            prompt = template
            prompt = prompt.replace('[title]', entry.title)
            prompt = prompt.replace('[description]', entry.description)
            prompt = prompt.replace('[category_name]', display_name)
            prompt = prompt.replace('[criteria]', category.criteria)
            prompt = prompt.replace('[examples]', '\n'.join(f'- {ex}' for ex in category.examples))
            prompt = prompt.replace('[values]', category.values)
            
            return prompt
            
        except Exception as e:
            print(f"Error constructing prompt: {str(e)}")
            raise

class ResultsManager:
    """Manages classification results and metrics"""
    
    def __init__(self):
        self.results = []

    def _transform_value(self, value: str) -> str:
        """Transform values for output"""
        # Convert German Ja/Nein to 1/0
        value = str(value).strip().lower()
        if value in ['ja', 'ja (1)', '(1)', '1']:
            return "1"
        elif value in ['nein', 'nein (0)', '(0)', '0']:
            return "0"
        # Return original value if no transformation needed
        return value

    async def save_results(self, results: List[ProcessingResult], output_base: str):
        """Save results in Excel format"""
        # Group results by title
        entries = {}
        categories = set()
        
        for result in results:
            if result.title not in entries:
                clean_description = ' '.join(
                    result.description
                    .replace('\n', ' ')
                    .split()
                )
                entries[result.title] = {
                    'title': result.title,
                    'description': clean_description
                }
            # Store AI code, confidence, and reasoning for each category
            cat = result.category
            entries[result.title][f'ai_{cat}'] = self._transform_value(result.ai_code)  # Now self is defined
            entries[result.title][f'confidence_{cat}'] = f"{result.confidence:.2f}"
            entries[result.title][f'reasoning_{cat}'] = result.reasoning
            categories.add(cat)
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(entries, orient='index')
        
        # Organize columns
        ai_cols = [f'ai_{cat}' for cat in sorted(categories)]
        confidence_cols = [f'confidence_{cat}' for cat in sorted(categories)]
        reasoning_cols = [f'reasoning_{cat}' for cat in sorted(categories)]
        
        columns = ['title', 'description'] + ai_cols + confidence_cols + reasoning_cols
        df = df.reindex(columns=columns)
        
        # Ensure results directory exists
        results_dir = os.path.join(root_dir, 'data', 'results')  # Use root_dir to get absolute path
        os.makedirs(results_dir, exist_ok=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(results_dir, f"results_{timestamp}.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"\nResults saved to Excel: {excel_path}")
        
        return excel_path
    
    @staticmethod
    async def calculate_metrics(results: List[ProcessingResult]) -> Dict:
        """Calculate basic statistics about AI classifications"""
        by_category = defaultdict(list)
        for r in results:
            by_category[r.category].append(r)
        
        metrics = {'by_category': {}, 'overall': {}}
        
        for category, category_results in by_category.items():
            ai_codes = [r.ai_code for r in category_results]
            confidence_scores = [r.confidence for r in category_results]
            
            metrics['by_category'][category] = {
                'total_samples': len(category_results),
                'positive_classifications': sum(1 for code in ai_codes if code == "1"),
                'average_confidence': sum(confidence_scores) / len(confidence_scores)
            }
        
        return metrics

# ============================================================================
# AI and Validation Components
# ============================================================================

class GPTClassificationInput(BaseModel):
    """Input structure for GPT classification"""
    prompt: str
    model: str
    temperature: float

class GPTClassificationOutput(BaseModel):
    """Output structure for GPT classification"""
    response: str

class ResponseValidator:
    """Validates and interprets GPT responses"""
    
    def validate_response(self, response: str, logger: logging.Logger) -> ValidationResult:
        """Validate and interpret GPT response in JSON format"""
        try:
            # Debug raw response
            logger.info(f"\n🔍 Raw GPT response:")
            logger.info(response)
            
            # Clean the response if it contains markdown code blocks
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()
            
            # Parse JSON response
            try:
                response_data = json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.error(f"Response content: {response[:200]}...")
                return ValidationResult(
                    value="0",  # Default to 0 on error
                    confidence=0.0,
                    confidence_level=ConfidenceLevel.LOW,
                    reasoning=f"Error parsing JSON response: {str(e)}"
                )
            
            logger.info(f"\n📋 Parsed JSON data:")
            logger.info(response_data)
            
            # Validate required fields
            required_fields = ['value', 'confidence', 'reasoning']
            missing_fields = [field for field in required_fields if field not in response_data]
            if missing_fields:
                logger.error(f"Missing required fields in response: {missing_fields}")
                return ValidationResult(
                    value="0",  # Default to 0 on error
                    confidence=0.0,
                    confidence_level=ConfidenceLevel.LOW,
                    reasoning=f"Missing required fields in response: {missing_fields}"
                )
            
            # Get value directly without conversion
            value = str(response_data.get('value', ''))  # Just convert to string
            
            # Debug output
            logger.info(f"\n🔄 Value from GPT: {value}")
            
            # Get confidence and reasoning
            try:
                confidence = float(response_data.get('confidence', 0.0))
                if confidence < 0.0 or confidence > 1.0:
                    logger.warning(f"Confidence value {confidence} out of range [0,1], clamping to nearest valid value")
                    confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                logger.error(f"Invalid confidence value: {response_data.get('confidence')}")
                confidence = 0.0
            
            reasoning = str(response_data.get('reasoning', ''))
            
            return ValidationResult(
                value=value,  # Keep original value
                confidence=confidence,
                confidence_level=self._get_confidence_level(confidence),
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Unexpected error in response validation: {str(e)}")
            return ValidationResult(
                value="0",  # Default to 0 on error
                confidence=0.0,
                confidence_level=ConfidenceLevel.LOW,
                reasoning=f"Error validating response: {str(e)}"
            )

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Helper to determine confidence level"""
        if confidence < 0.33:
            return ConfidenceLevel.LOW
        elif confidence < 0.67:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.HIGH

# GPT agent handles the AI interaction
class GPTClassificationAgent:
    """GPT agent for classifying training data entries"""
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=120.0  # Increase timeout to 120 seconds
        )
        self.logger = logging.getLogger('gpt_agent')  # Get a logger for GPT agent

    async def process(self, input_data: GPTClassificationInput) -> GPTClassificationOutput:
        try:
            # Debug output - safer parsing
            self.logger.info("\n🔍 Sending to GPT:")
            self.logger.info("=" * 50)
            try:
                # Extract title from prompt
                prompt_lines = input_data.prompt.split('\n')
                title = next((line.split('Titel: ')[1] for line in prompt_lines 
                            if 'Titel: ' in line), 'No title found')
                
                # Extract category from JSON template
                category = next((line.split('category": "')[1].split('"')[0] 
                               for line in prompt_lines if '"category": "' in line), 
                              'No category found')
                
                self.logger.info(f"Title: {title}")
                self.logger.info(f"Category: {category}")
            except Exception as e:
                self.logger.error(f"Error parsing prompt for debug output: {str(e)}")
                self.logger.info("Raw prompt:")
                self.logger.info(input_data.prompt)
            
            self.logger.info("-" * 50)
            self.logger.info("Full Prompt:")
            self.logger.info(input_data.prompt)
            self.logger.info("=" * 50)

            try:
                # Add response format specification to ensure JSON output
                response = self.client.chat.completions.create(
                    model=input_data.model,
                    temperature=input_data.temperature,
                    messages=[
                        {
                            "role": "system", 
                            "content": "Du bist ein wissenschaftlicher Coder, spezialisiert auf strukturierte Daten. Bitte antworte immer im JSON-Format mit den Feldern 'value', 'confidence' und 'reasoning'."
                        },
                        {
                            "role": "user",
                            "content": input_data.prompt + "\n\nBitte antworte im folgenden JSON-Format:\n{\n  \"value\": \"0\" oder \"1\",\n  \"confidence\": Zahl zwischen 0 und 1,\n  \"reasoning\": \"Deine Begründung\"\n}"
                        }
                    ],
                    max_tokens=500,  # Limit response length
                    timeout=120.0  # Timeout in seconds
                )

                # Check if we got a valid response
                if not response.choices:
                    raise Exception("No response received from GPT")
                
                response_content = response.choices[0].message.content
                if not response_content:
                    raise Exception("Empty response from GPT")

                # Check for HTML in response
                if response_content.strip().startswith('<'):
                    raise Exception("Received HTML response instead of JSON. This might indicate a server error or timeout.")

                # Debug raw response
                self.logger.info("\n📝 Raw GPT Response:")
                self.logger.info("=" * 50)
                self.logger.info(response_content)
                self.logger.info("=" * 50)

                # Clean the response if it contains markdown code blocks
                if '```json' in response_content:
                    response_content = response_content.split('```json')[1].split('```')[0].strip()
                elif '```' in response_content:
                    response_content = response_content.split('```')[1].split('```')[0].strip()

                # Validate that the response is JSON
                try:
                    json.loads(response_content)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON response: {str(e)}")
                    self.logger.error(f"Response content: {response_content[:500]}...")
                    raise Exception(f"GPT response is not valid JSON. Response content: {response_content[:200]}...")

                return GPTClassificationOutput(response=response_content)

            except Exception as e:
                self.logger.error(f"\n❌ Error in GPT request: {str(e)}")
                self.logger.error("Request details:")
                self.logger.error(f"Model: {input_data.model}")
                self.logger.error(f"Temperature: {input_data.temperature}")
                self.logger.error(f"API Key (first 8 chars): {api_key[:8]}...")
                # Add more detailed error information
                if hasattr(e, 'response'):
                    self.logger.error(f"Response status: {e.response.status_code}")
                    self.logger.error(f"Response body: {e.response.text}")
                raise

        except Exception as e:
            self.logger.error(f"Error in GPT processing: {str(e)}")
            # Add more context to the error
            if "SSL" in str(e):
                self.logger.error("SSL connection error detected. This might be due to:")
                self.logger.error("1. Network connectivity issues")
                self.logger.error("2. SSL certificate verification problems")
                self.logger.error("3. Proxy or firewall settings")
                self.logger.error("4. API key authentication issues")
            raise

# ============================================================================
# YAML Management Components
# ============================================================================

class YAMLManager:
    """
    Manages YAML operations for the coding scheme
    
    Responsibilities:
    1. Generate YAML from DOCX
    2. Fix YAML formatting
    3. Validate YAML structure
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger("yaml_manager")
    
    def parse_condition(self, values: str) -> dict:
        """Parse condition from values string"""
        if "wenn" not in values.lower():
            return None
            
        condition = values.lower().split("wenn")[1].strip()
        
        if "min. eine der kategorien" in condition:
            try:
                range_part = condition.split("kategorien")[1].split("=")[0].strip()
                start, end = range_part.split("-")
                value = condition.split("=")[1].strip()
                return {
                    "type": "any_in_range",
                    "range_start": start.strip(),
                    "range_end": end.strip(),
                    "value": value
                }
            except Exception as e:
                self.logger.warning(f"Could not parse range condition: {condition}")
                return None
        elif "=" in condition:
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
        base_values = values.split("wenn")[0].strip() if "wenn" in values.lower() else values
        
        if "ja" in base_values.lower() and "nein" in base_values.lower():
            return "Ja (1), Nein (0)"
        elif "offen" in base_values.lower():
            return "offen"
        elif any(x in base_values for x in [";", ","]):
            return base_values.replace("\n", " ").strip()
        else:
            return base_values

    def simplify_category_name(self, name: str) -> str:
        """Remove numbering pattern from category names"""
        # Check if it's a derived category
        is_derived = name.startswith('_DERIVED_')
        base_name = name[9:] if is_derived else name  # Remove _DERIVED_ prefix if present
        
        patterns = [
            r'^\d+\.\d+\.?\d*\s*[a-z]?\s*',
            r'^\.\d+\s*',
            r'^\d+\s+'
        ]
        
        result = base_name
        for pattern in patterns:
            result = re.sub(pattern, '', result)
        
        # Add back the _DERIVED_ prefix if it was present
        if is_derived:
            result = '_DERIVED_' + result
        
        return result.strip()

    def fix_criteria(self, criteria: str) -> str:
        """Fix formatting of criteria text"""
        criteria = re.sub(r'^Kriterium:\s*', '', criteria)
        criteria = criteria.replace('"', "'").strip()
        return criteria

    def fix_examples(self, examples: list) -> list:
        """Fix formatting of examples"""
        fixed = []
        for example in examples:
            if isinstance(example, dict):
                example = "; ".join(f"{k}: {v}" for k, v in example.items())
            
            example = re.sub(r'^[·•]\s*', '', str(example))
            example = example.strip().replace('"', "'")
            
            if not (example.startswith('"') and example.endswith('"')):
                example = f'"{example}"'
            fixed.append(example)
        return fixed

    def fix_values(self, values: str) -> str:
        """Fix formatting of values"""
        if not values:
            return '""'
            
        values = str(values).strip().strip('"\'')
        
        # Remove trailing commas and semicolons
        values = values.rstrip(',;')
        
        # Handle special cases
        if values.lower() == "offen":
            return '"offen"'
            
        # Handle derived categories
        if values.lower() == "ja (1)":
            return '"Ja (1)"'
            
        # Handle conditional values
        if "wenn" in values.lower():
            return f'"{values}"'
            
        # Handle standard yes/no format
        if "ja" in values.lower() and "nein" in values.lower():
            return '"Ja (1), Nein (0)"'
            
        # Handle values with semicolons or commas
        if ";" in values or "," in values:
            # Split by either semicolon or comma
            parts = []
            for part in values.replace(';', ',').split(','):
                part = part.strip()
                if part:  # Only add non-empty parts
                    parts.append(part)
            # Join with consistent separator
            return f'"{", ".join(parts)}"'
            
        # Default case: wrap in quotes if not already
        if not values.startswith('"'):
            values = f'"{values}"'
            
        return values

    async def generate_yaml_from_docx(self) -> bool:
        """Generate YAML from DOCX file"""
        try:
            # Get the directory containing the coding scheme file
            doc_dir = os.path.dirname(self.config['paths']['coding_scheme'])
            # Use the path from config
            input_file = os.path.join(doc_dir, self.config['paths']['docx_file'])
            output_file = os.path.join(doc_dir, "coding_scheme_imported.yml")
            
            self.logger.info(f"Generating YAML from DOCX: {input_file}")
            
            # Check if file exists
            if not os.path.exists(input_file):
                self.logger.error(f"DOCX file not found at: {input_file}")
                return False
            
            # Load DOCX
            doc = Document(input_file)
            table = doc.tables[0]
            codes = {}
            
            # Process table
            for row in table.rows[1:]:
                cells = row.cells
                category = cells[0].text.strip()
                values = cells[1].text.strip()
                criteria = cells[2].text.strip()
                examples = cells[3].text.strip()
                
                condition = self.parse_condition(values)
                if condition:
                    category = "_DERIVED_" + category
                
                codes[category] = {
                    "criteria": criteria,
                    "examples": examples.split('\n') if examples else [''],
                    "values": self.standardize_values(values)
                }
                if condition:
                    codes[category]["condition"] = condition
            
            # Save YAML
            with open(output_file, "w", encoding="utf-8") as yaml_file:
                yaml.dump(codes, yaml_file, allow_unicode=True, default_flow_style=False)
            
            self.logger.info(f"Generated YAML saved to: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating YAML: {str(e)}")
            return False

    async def fix_yaml_format(self) -> bool:
        """Fix YAML formatting"""
        try:
            # Get the directory containing the coding scheme file
            doc_dir = os.path.dirname(self.config['paths']['coding_scheme'])
            input_file = os.path.join(doc_dir, "coding_scheme_imported.yml")
            output_file = self.config['paths']['coding_scheme']
            
            self.logger.info(f"Fixing YAML format: {input_file}")
            
            # Read original YAML
            with open(input_file, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
            
            # Fix format - only include non-derived categories
            fixed_data = {}
            for key, value in data.items():
                # Skip _DERIVED_ categories completely
                if key.startswith('_DERIVED_'):
                    self.logger.info(f"Skipping derived category: {key}")
                    continue
                
                # Fix content for non-derived categories
                new_key = self.simplify_category_name(key)
                fixed_content = {
                    'criteria': self.fix_criteria(value['criteria']),
                    'examples': self.fix_examples(value['examples']),
                    'values': self.fix_values(value.get('values', ''))
                }
                
                fixed_data[new_key] = fixed_content
                
                # Log the conversion
                self.logger.info(f"Converted category:\n  From: {key}\n  To:   {new_key}\n  Values: {fixed_content['values']}")
            
            # Save fixed YAML with proper formatting
            with open(output_file, 'w', encoding='utf-8') as file:
                yaml.dump(fixed_data, file, 
                         allow_unicode=True, 
                         sort_keys=False, 
                         default_flow_style=False,
                         indent=2,
                         width=1000)  # Increase width to prevent line wrapping
            
            self.logger.info(f"Fixed YAML saved to: {output_file}")
            self.logger.info("Categories included in YAML:")
            for category in sorted(fixed_data.keys()):
                self.logger.info(f"- {category}")
            
            # Verify the YAML is valid
            try:
                with open(output_file, 'r', encoding='utf-8') as file:
                    yaml.safe_load(file)
                self.logger.info("YAML validation successful")
            except Exception as e:
                self.logger.error(f"YAML validation failed: {str(e)}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error fixing YAML format: {str(e)}")
            return False

    async def validate_yaml(self) -> bool:
        """Validate YAML structure"""
        try:
            self.logger.info(f"Validating YAML: {self.config['paths']['coding_scheme']}")
            
            with open(self.config['paths']['coding_scheme'], 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
            
            # Validate each category
            for category_name, category_data in data.items():
                if not isinstance(category_data, dict):
                    self.logger.error(f"Invalid category structure: {category_name}")
                    return False
                
                required_fields = ['criteria', 'examples', 'values']
                for field in required_fields:
                    if field not in category_data:
                        self.logger.error(f"Missing field '{field}' in category: {category_name}")
                        return False
                
                if not isinstance(category_data['examples'], list):
                    self.logger.error(f"'examples' must be a list in category: {category_name}")
                    return False
            
            self.logger.info("YAML validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating YAML: {str(e)}")
            return False

    async def update_coding_scheme(self) -> bool:
        """
        Update coding scheme by running the complete YAML pipeline:
        1. Generate YAML from DOCX
        2. Fix YAML format
        3. Validate final YAML
        """
        try:
            self.logger.info("Starting coding scheme update...")
            
            # Generate YAML from DOCX
            if not await self.generate_yaml_from_docx():
                self.logger.error("Failed to generate YAML from DOCX")
                return False
            
            # Fix YAML format
            if not await self.fix_yaml_format():
                self.logger.error("Failed to fix YAML format")
                return False
            
            # Validate final YAML
            if not await self.validate_yaml():
                self.logger.error("Failed to validate YAML")
                return False
            
            self.logger.info("Coding scheme update completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating coding scheme: {str(e)}")
            return False

# ============================================================================
# Classification Coordinator
# ============================================================================

class TrainingDataClassifier:
    """
    Main pipeline coordinator for training data classification
    
    Process Flow:
    1. Load and validate input data:
       - Training data CSV
       - Human codes (optional)
       - Coding scheme YAML
       - Prompt template
       
    2. For each training entry:
       - Process each category in coding scheme
       - Generate category-specific prompts
       - Get GPT classification
       - Validate and interpret responses
       
    3. Output handling:
       - Save results to CSV
       - Calculate agreement metrics
       - Display evaluation results
    
    Components:
    - DataManager: Handles data loading and processing
    - ResourceManager: Manages coding scheme and prompts
    - GPTClassificationAgent: Handles AI interaction
    - ResponseValidator: Validates GPT responses
    - ResultsManager: Handles results and metrics
    - YAMLManager: Manages YAML operations
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger("training_classifier")
        
        # Initialize components
        self.data_manager = DataManager()
        self.resource_manager = ResourceManager()
        self.results_manager = ResultsManager()
        self.classification_agent = GPTClassificationAgent()
        self.response_validator = ResponseValidator()
        self.yaml_manager = YAMLManager(config)  # Add YAML manager

    async def process_entries(self, entries: List[DataEntry], scheme: CodingScheme) -> None:
        """Process all entries through the pipeline"""
        print("\nStarting pipeline processing...")
        
        # Load template once
        template = await self.resource_manager.load_template(self.config['paths']['prompt_template'])
        
        # Process each entry
        for entry in entries:
            results = await self.process_entry(entry, template, scheme)
            
            # Store results
            for result in results:
                await self.results_manager.store_result(
                    entry=entry,
                    category=result.category,
                    value=result.value,
                    confidence=result.confidence
                )

    async def process_entry(self, entry: DataEntry, template: str, scheme: CodingScheme) -> List[ProcessingResult]:
        """Process a single entry for selected categories"""
        results = []
        
        print(f"\n📊 Processing Entry:")
        print(f"Title: {entry.title}")
        print(f"Description: {entry.description[:100]}...")
        
        # Use categories from config instead of hardcoded list
        selected_categories = self.config.get('selected_categories', [])
        if not selected_categories:
            print("Warning: No categories selected in config")
            return results
        
        # Process each category only once
        processed_categories = set()
        for category_key in selected_categories:
            if category_key in processed_categories:
                continue
                
            if category_key not in scheme.categories:
                print(f"Warning: Category {category_key} not found in scheme")
                continue
            
            print(f"\n📋 Category: {category_key}")
            processed_categories.add(category_key)
            
            try:
                # Generate prompt
                prompt = await self.resource_manager.construct_prompt(
                    template, entry, scheme, category_key
                )
                
                # Create GPT input
                gpt_input = GPTClassificationInput(
                    prompt=prompt,
                    model=self.config['gpt']['model'],
                    temperature=self.config['gpt']['temperature']
                )
                
                # Get and validate classification
                gpt_output = await self.classification_agent.process(gpt_input)
                validation_result = self.response_validator.validate_response(
                    gpt_output.response,
                    logger=self.logger
                )
                
                # Add to results
                results.append(ProcessingResult(
                    title=entry.title,
                    description=entry.description,
                    category=category_key,
                    ai_code=validation_result.value,  # Use the value as-is
                    confidence=validation_result.confidence,
                    reasoning=validation_result.reasoning or ""
                ))
                
                print(f"Result: {validation_result.value} (confidence: {validation_result.confidence:.2f})")
                
            except Exception as e:
                print(f"Error processing category {category_key}: {str(e)}")
                continue
                
        return results

    async def run(self):
        """Run the complete classification process"""
        self.logger.info("Starting classification")
        try:
            # Check if we have temporary files to use
            if self.config.get('temp_files', {}).get('data_csv'):
                self.config['paths']['data_csv'] = self.config['temp_files']['data_csv']
            if self.config.get('temp_files', {}).get('coding_scheme'):
                self.config['paths']['coding_scheme'] = self.config['temp_files']['coding_scheme']
            if self.config.get('temp_files', {}).get('prompt_template'):
                self.config['paths']['prompt_template'] = self.config['temp_files']['prompt_template']

            # Generate YAML from DOCX if needed
            if not os.path.exists(self.config['paths']['coding_scheme']):
                self.logger.info("Found DOCX file, updating coding scheme...")
                if not await self.yaml_manager.update_coding_scheme():
                    self.logger.error("Failed to update coding scheme")
                    return False
            
            # Load and validate resources
            dataset = await self.data_manager.load_data(self.config['paths']['data_csv'])
            if self.config['paths'].get('human_codes'):
                codes = await self.data_manager.load_data(self.config['paths']['human_codes'])
                dataset = await self.data_manager.merge_datasets(dataset, codes)
            
            # Load scheme
            scheme = await self.resource_manager.load_scheme(self.config['paths']['coding_scheme'])
            template = await self.resource_manager.load_template(self.config['paths']['prompt_template'])
            
            # Process entries
            all_results = []
            entry_count = 0
            
            # Get selected categories
            selected_categories = self.config.get('selected_categories', [])
            if not selected_categories:
                self.logger.error("No categories selected in config")
                return False
                
            # Process each entry once
            for _, row in dataset.iterrows():
                entry = DataEntry(
                    title=row["title"],
                    description=row["description"],
                    human_code="0"  # Default value
                )
                entry_count += 1
                self.logger.info(f"Processing entry {entry_count}: {entry.title}")
                
                # Process all selected categories for this entry
                results = await self.process_entry(entry, template, scheme)
                all_results.extend(results)
            
            # Save results with timestamp
            output_path = await self.results_manager.save_results(
                all_results, 
                self.config['paths']['output_base']
            )
            self.logger.info(f"Results saved to: {output_path}")
            
            self.logger.info("Pipeline completed successfully")
            return True
            
        except KeyboardInterrupt:
            print("\n\n🛑 Process interrupted by user")
            print("Cleaning up and shutting down...")
            return False
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            print("\n✨ Pipeline finished")

    async def validate_data_consistency(self):
        """Validate consistency between data files"""
        try:
            # Load files
            training_data = await self.data_manager.load_data(self.config['paths']['data_csv'])
            human_codes = await self.data_manager.load_data(self.config['paths']['human_codes'])
            scheme = await self.resource_manager.load_scheme(self.config['paths']['coding_scheme'])
            
            # Check column presence
            print("\nValidating data consistency:")
            print("1. Training data columns:", training_data.columns.tolist())
            print("2. Human codes columns:", human_codes.columns.tolist())
            print("3. Categories in scheme:", list(scheme.categories.keys()))
            
            # Check matching entries
            training_titles = set(training_data['title'])
            human_code_titles = set(human_codes['title'])
            if training_titles != human_code_titles:
                print("\n⚠️ Warning: Mismatched titles between training data and human codes")
                print("Missing in human codes:", training_titles - human_code_titles)
                print("Extra in human codes:", human_code_titles - training_titles)
            
            return True
        except Exception as e:
            print(f"\n❌ Data consistency error: {str(e)}")
            return False

async def main():
    """Main entry point for the classification pipeline"""
    # Check for API key before proceeding
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set. Please check your .env file.")
        sys.exit(1)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory structure
    os.makedirs(CONFIG['paths']['log_dir'], exist_ok=True)
    os.makedirs(CONFIG['paths']['output_dir'], exist_ok=True)
    
    # Create run-specific log file with rotation
    log_file = os.path.join(CONFIG['paths']['log_dir'], f'pipeline_run_{timestamp}.log')
    
    # Setup logging with custom formatting and rotation
    formatter = logging.Formatter(
        json.dumps(CONFIG['logging']['format']),
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=CONFIG['logging']['max_bytes'],
        backupCount=CONFIG['logging']['backup_count'],
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, CONFIG['logging']['level']))
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Get our specific logger
    logger = logging.getLogger('pipeline_run')
    
    try:
        # Log configuration details
        logger.info("Pipeline Run Configuration", extra={
            'timestamp': timestamp,
            'gpt_model': CONFIG['gpt']['model'],
            'temperature': CONFIG['gpt']['temperature']
        })
        
        # Log complete coding scheme
        logger.info("Complete Coding Scheme", extra={
            'timestamp': timestamp,
            'coding_scheme_path': CONFIG['paths']['coding_scheme']
        })
        
        with open(CONFIG['paths']['coding_scheme'], 'r', encoding='utf-8') as f:
            scheme = yaml.safe_load(f)
            scheme_str = yaml.dump(scheme, allow_unicode=True, default_flow_style=False)
            logger.info("Coding Scheme Structure", extra={
                'scheme': scheme_str,
                'categories': list(scheme.keys())
            })
        
        # Check required files
        required_files = [
            CONFIG['paths']['data_csv'],
            CONFIG['paths']['human_codes'],
            CONFIG['paths']['coding_scheme'],
            CONFIG['paths']['prompt_template']
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            logger.error("Missing required files", extra={'missing_files': missing_files})
            return
        
        # Create and run classifier
        classifier = TrainingDataClassifier(CONFIG)
        await classifier.run()
        
        logger.info("Pipeline Run Completed", extra={'timestamp': datetime.now().isoformat()})
        
    except Exception as e:
        logger.error("Pipeline failed", extra={
            'error': str(e),
            'traceback': sys.exc_info()
        })
    finally:
        # Log final timestamp
        logger.info("Run finished", extra={'timestamp': datetime.now().isoformat()})

if __name__ == "__main__":
    asyncio.run(main())
