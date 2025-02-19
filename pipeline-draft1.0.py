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
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
import sys
from collections import defaultdict

load_dotenv()  # Add this at the top of the file

# Configuration for file paths, logging, and GPT settings
CONFIG = {
    'paths': {
        'data_csv': "training_data.xlsx",        # Changed from CSV to Excel
        'human_codes': "human_codes.xlsx",              
        'coding_scheme': "coding_scheme.yml",           
        'prompt_template': "prompt.txt",                
        'output_dir': "results",                        
        'log_dir': "log",  
        'output_base': "results/ai_coded_results"       
    },
    'logging': {
        'level': 'INFO',
        'file': os.path.join('logs', 'atomic_agents.log')
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
    criteria: str
    examples: List[str]
    values: str

class CodingScheme(BaseModel):
    """Structure for coding scheme"""
    categories: Dict[str, CodingSchemeCategory]

    @classmethod
    def from_yaml(cls, data: Dict) -> 'CodingScheme':
        """Create CodingScheme from YAML data"""
        categories = {
            key: CodingSchemeCategory(**value) 
            for key, value in data.items()
        }
        return cls(categories=categories)

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
    """
    Manages loading and preparation of classification resources
    
    Responsibilities:
    1. Load and validate coding scheme from YAML
    2. Load prompt template
    3. Construct category-specific prompts
    """
    @staticmethod
    async def load_scheme(path: str) -> CodingScheme:
        """
        Load and validate coding scheme from YAML
        
        Args:
            path: Path to coding scheme YAML file
            
        Returns:
            Validated CodingScheme
            
        Raises:
            FileNotFoundError: If scheme file doesn't exist
            ValueError: If scheme structure is invalid
        """
        try:
            with open(path, "r", encoding="utf-8") as file:
                raw_scheme = yaml.safe_load(file)
                return CodingScheme.from_yaml(raw_scheme)
        except FileNotFoundError:
            raise FileNotFoundError(f"Coding scheme file not found: {path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in coding scheme: {str(e)}")
    
    @staticmethod
    async def load_template(path: str) -> str:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    
    @staticmethod
    async def construct_prompt(template: str, entry: DataEntry, scheme: CodingScheme, category_key: str) -> str:
        """Construct category-specific prompt"""
        try:
            print(f"\nConstructing prompt for YAML category: {category_key}")
            if not category_key in scheme.categories:
                raise ValueError(f"Category {category_key} not found in coding scheme")
            
            category = scheme.categories[category_key]
            prompt = template.replace('[title]', entry.title)
            prompt = prompt.replace('[description]', entry.description)
            prompt = prompt.replace('[category_name]', category_key)  # Use exact YAML key
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
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = f"{output_base}_{timestamp}.xlsx"
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
            print(f"\n🔍 Raw GPT response:")
            print(response)
            
            # Parse JSON response
            response_data = json.loads(response)
            print(f"\n📋 Parsed JSON data:")
            print(response_data)
            
            # Get value directly without conversion
            value = str(response_data.get('value', ''))  # Just convert to string
            
            # Debug output
            print(f"\n🔄 Value from GPT: {value}")
            
            # Get confidence and reasoning
            confidence = float(response_data.get('confidence', 0.0))
            reasoning = str(response_data.get('reasoning', ''))
            
            return ValidationResult(
                value=value,  # Keep original value
                confidence=confidence,
                confidence_level=self._get_confidence_level(confidence),
                reasoning=reasoning
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            return ValidationResult(
                value=response,  # Return raw response if JSON parsing fails
                confidence=0.0,
                confidence_level=ConfidenceLevel.LOW,
                reasoning=f"Error parsing JSON response: {str(e)}"
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
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

            response = self.client.chat.completions.create(
                model=input_data.model,
                temperature=input_data.temperature,
                messages=[
                    {
                        "role": "system", 
                        "content": "Du bist ein wissenschaftlicher Coder, spezialisiert auf strukturierte Daten."
                    },
                    {"role": "user", "content": input_data.prompt}
                ]
            )

            # Check if we got a valid response
            if not response.choices:
                raise Exception("No response received from GPT")
            
            response_content = response.choices[0].message.content
            if not response_content:
                raise Exception("Empty response from GPT")

            # Debug output for response
            self.logger.info("\n📝 GPT Response:")
            self.logger.info("=" * 50)
            self.logger.info(response_content)
            self.logger.info("=" * 50)

            return GPTClassificationOutput(response=response_content)

        except Exception as e:
            self.logger.error(f"\n❌ Error in GPT request: {str(e)}")
            self.logger.error("Request details:")
            self.logger.error(f"Model: {input_data.model}")
            self.logger.error(f"Temperature: {input_data.temperature}")
            self.logger.error(f"API Key (first 8 chars): {os.getenv('OPENAI_API_KEY')[:8]}...")
            raise

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

    # Keep the SELECTED_CATEGORIES
    SELECTED_CATEGORIES = [

        ## Basic Categories
        # "Kursname",
        # "Anbieter",
        # "Name der Hochschule",
        # "Annehmerhochschulen des Hochschulanbieters",
        # "Anzahl",
        # "Dozent_in",
        # "Veranstaltungsart",
        # "Format der Veranstaltung Präsenz/Online",
        # "Format der Veranstaltung synchron",
        # "Stattfinden der Veranstaltung",
        # "Termin der Veranstaltung",
        # "Dauer",
        # "Semester",
        # "Zielgruppe",
        "Fachbereich",
        # "Fachbereich offen",
        # "Kompetenzbereich des Zertifikats Hochschullehre",
        # "Pädagogisch-didaktisches Angebot allgemeiner oder fachbereichsspezifischer Art",
        
        ## Digital Competence Categories
        # "Organisatorische Kommunikation",
        # "Berufliche Zusammenarbeit",
        # "Reflektierte Praxis",
        # "Digitale fortlaufende berufliche Entwicklung",
        # "Auswahl digitaler Ressourcen",
        # "Erstellen und Anpassen digitaler Ressourcen",
        # "Organisieren, Schützen und Teilen digitaler Ressourcen",
        # "Lehren",
        # "Lernbegleitung",
        # "Kollaboratives Lernen",
        # "Selbstgesteuertes Lernen",
        # "Lernstand erheben",
        # "Lern-Evidenzen analysieren",
        # "Feedback und Planung",
        # "Digitale Teilhabe",
        # "Differenzierung und Individualisierung",
        # "Aktive Einbindung von Lernenden",
        "Förderung der Informations- und Medienkompetenz der Studierenden",
        "Förderung der Digitalen Kommunikation und Zusammenarbeit der Studierenden",
        # "Förderung der Erstellung digitaler Inhalte der Studierenden",
        # "Förderung des verantwortungsvollen Umgang der Studierenden mit digitalen Medien",
        # "Förderung der Studierenden zum Digitalen Problemlösen",
        "Informations- und Medienkompetenz",
        "Digitale Kommunikation und Zusammenarbeit",
        # "Erstellung digitaler Inhalte",
        # "Verantwortungsvoller Umgang mit digitalen Medien",
        # "Digitales Problemlösen"
    ]

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
        
        for category_key in self.SELECTED_CATEGORIES:
            if category_key not in scheme.categories:
                print(f"Warning: Category {category_key} not found in scheme")
                continue
            
            print(f"\n📋 Category: {category_key}")
            
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
            # Load and validate resources
            dataset = await self.data_manager.load_data(self.config['paths']['data_csv'])
            if self.config['paths'].get('human_codes'):
                codes = await self.data_manager.load_data(self.config['paths']['human_codes'])
                dataset = await self.data_manager.merge_datasets(dataset, codes)
            
            # Load scheme
            scheme = await self.resource_manager.load_scheme(self.config['paths']['coding_scheme'])
            template = await self.resource_manager.load_template(self.config['paths']['prompt_template'])
            
            # Process each category
            all_results = []
            categories = list(scheme.categories.keys())
            if self.config.get('test_mode', {}).get('enabled', False):
                categories = categories[:self.config['test_mode']['max_categories']]
            
            for category in categories:
                entry_count = 0
                async for entry in self.data_manager.iterate_entries(dataset, category):
                    entry_count += 1
                    self.logger.info(f"Processing entry {entry_count} for category {category}")
                    results = await self.process_entry(entry, template, scheme)
                    all_results.extend(results)
            
            # Save results with timestamp
            output_path = await self.results_manager.save_results(
                all_results, 
                self.config['paths']['output_base']
            )
            self.logger.info(f"Results saved to: {output_path}")
            
            # Skip metrics for now
            # metrics = await self.results_manager.calculate_metrics(all_results)
            
            self.logger.info("Pipeline completed successfully")
            
        except KeyboardInterrupt:
            print("\n\n🛑 Process interrupted by user")
            print("Cleaning up and shutting down...")
            # Add any cleanup code here
            return
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory structure
    os.makedirs('log', exist_ok=True)
    os.makedirs(CONFIG['paths']['output_dir'], exist_ok=True)
    
    # Create run-specific log file
    log_file = os.path.join('log', f'pipeline_run_{timestamp}.log')
    
    # Setup logging with custom formatting
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)  # Explicitly use stdout
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Get our specific logger
    logger = logging.getLogger('pipeline_run')
    
    try:
        # Log configuration details
        logger.info("=== Pipeline Run Configuration ===")
        logger.info(f"Timestamp: {timestamp}")
        logger.info(f"GPT Model: {CONFIG['gpt']['model']}")
        logger.info(f"Temperature: {CONFIG['gpt']['temperature']}")
        
        # Log complete coding scheme
        logger.info("\n=== APPENDIX: Complete Coding Scheme ===")
        with open(CONFIG['paths']['coding_scheme'], 'r', encoding='utf-8') as f:
            scheme = yaml.safe_load(f)
            scheme_str = yaml.dump(scheme, allow_unicode=True, default_flow_style=False)
            logger.info("\nCoding Scheme Structure:")
            logger.info("-" * 80)
            for line in scheme_str.split('\n'):
                logger.info(line)
            logger.info("-" * 80)
            
            # Also log summary of categories
            logger.info("\nCategories Summary:")
            for category in scheme.keys():
                logger.info(f"- {category}")
        
        # Check required files
        required_files = [
            CONFIG['paths']['data_csv'],
            CONFIG['paths']['human_codes'],
            CONFIG['paths']['coding_scheme'],
            CONFIG['paths']['prompt_template']
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            logger.error("Missing required files:")
            for file in missing_files:
                logger.error(f"- {file}")
            return
        
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not found in environment variables")
            return
        
        # Create and run classifier
        classifier = TrainingDataClassifier(CONFIG)
        await classifier.run()
        
        logger.info("=== Pipeline Run Completed ===")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
    finally:
        # Log final timestamp
        logger.info(f"Run finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main())
