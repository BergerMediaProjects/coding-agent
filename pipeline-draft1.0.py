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
from typing import Dict, List, AsyncGenerator, Optional
from pydantic import BaseModel, Field
import asyncio
import logging
from atomic_agents import (
    AtomicAgent,
    AgentContext,
    AgentInput,
    AgentOutput,
    AgentError
)
from datetime import datetime
from enum import Enum

# Configuration for file paths, logging, and GPT settings
CONFIG = {
    'paths': {
        'data_csv': "teacher_training_data.csv",        # Input training data
        'human_codes': "human_codes.csv",               # Human-assigned codes for comparison
        'coding_scheme': "coding_scheme.yml",           # Category definitions and criteria
        'prompt_template': "prompt.txt",                # Template for GPT prompts
        'output_base': "ai_coded_results"               # Base name for output files
    },
    'logging': {
        'level': 'INFO',
        'file': os.path.join('logs', 'atomic_agents.log')
    },
    'gpt': {
        'model': 'gpt-4',                              # GPT model to use
        'temperature': 0.0                             # 0.0 for most consistent results
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
    """
    Structure for storing classification results
    
    Fields:
    - title: Training entry title
    - description: Training description
    - category: Category from coding scheme being evaluated
    - human_code: Human-assigned classification (0 or 1)
    - ai_code: GPT-assigned classification (0 or 1)
    - confidence: GPT's confidence in classification (0.0 to 1.0)
    - reasoning: GPT's explanation for classification
    """
    title: str
    description: str
    category: str = Field(..., description="Category being evaluated")
    human_code: str = Field(..., pattern="^[01]$", description="Human-assigned code (0 or 1)")
    ai_code: str = Field(..., pattern="^[01]$", description="AI-assigned code (0 or 1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="AI confidence score")
    reasoning: str = Field(..., description="AI reasoning for the classification")

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
    value: str = Field(..., pattern="^[01]$", description="Classification value (0 or 1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Raw confidence score")
    confidence_level: ConfidenceLevel = Field(..., description="Interpreted confidence level")
    reasoning: str = Field(..., description="Classification reasoning")

class CodingSchemeCategory(BaseModel):
    """
    Structure for a single category in the coding scheme
    
    Fields:
    - criteria: Description of what qualifies for this category
    - examples: List of example cases that would be coded as 1
    - values: String describing the coding values (typically 'Ja (1), Nein (0)')
    """
    criteria: str = Field(..., description="Criteria for coding this category")
    examples: List[str] = Field(..., description="Example cases for value 1")
    values: str = Field(..., description="Allowed values (e.g., 'Ja (1), Nein (0)')")

class CodingScheme(BaseModel):
    """Complete coding scheme structure with validation"""
    categories: Dict[str, CodingSchemeCategory]

    @classmethod
    def from_yaml(cls, yaml_dict: Dict) -> 'CodingScheme':
        """
        Create validated scheme from raw YAML dict
        
        Args:
            yaml_dict: Raw dictionary from YAML file
            
        Returns:
            Validated CodingScheme
            
        Raises:
            ValueError: If scheme structure is invalid
        """
        try:
            return cls(categories={
                category_name: CodingSchemeCategory(**category_data)
                for category_name, category_data in yaml_dict.items()
            })
        except Exception as e:
            raise ValueError(f"Invalid coding scheme structure: {str(e)}")

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
        """Load and validate CSV data file"""
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Data file is empty: {path}")
        except Exception as e:
            raise ValueError(f"Error loading data file: {str(e)}")
    
    @staticmethod
    async def merge_datasets(main_df: pd.DataFrame, codes_df: pd.DataFrame) -> pd.DataFrame:
        return main_df.merge(codes_df, how='left', on='title')
    
    @staticmethod
    async def iterate_entries(df: pd.DataFrame) -> AsyncGenerator[DataEntry, None]:
        for _, row in df.iterrows():
            yield DataEntry(
                title=row["title"],
                description=row["description"],
                human_code=row.get("human_code", "Unknown")
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
        """
        Constructs prompt for a specific category from the coding scheme
        
        Process:
        1. Get category details from scheme
        2. Replace placeholders in template:
           - [title] with entry title
           - [description] with entry description
           - [category_name] with category key
           - [criteria] with category criteria
           - [values] with allowed values
           - [examples] with formatted example list
        
        Args:
            template: Base prompt template
            entry: Training data entry
            scheme: Validated coding scheme
            category_key: Category to evaluate
            
        Returns:
            Formatted prompt for GPT
        """
        # Get the category details from validated scheme
        category_data = scheme.categories[category_key]
        
        # Replace basic placeholders
        prompt = template.replace("[title]", entry.title)
        prompt = prompt.replace("[description]", entry.description)
        
        # Replace scheme-specific placeholders
        prompt = prompt.replace("[category_name]", category_key)
        prompt = prompt.replace("[criteria]", category_data.criteria)
        prompt = prompt.replace("[values]", category_data.values)
        
        # Format examples list with proper indentation
        examples_text = "\n".join(f"- {example}" for example in category_data.examples)
        prompt = prompt.replace("[examples]", examples_text)
        
        return prompt

class ResultsManager:
    """
    Manages classification results and metrics
    
    Responsibilities:
    1. Save results to timestamped CSV files
    2. Calculate agreement metrics:
       - Overall metrics
       - Per-category metrics
       - Cohen's Kappa scores
    """
    @staticmethod
    def get_timestamped_filename(base_name: str) -> str:
        """Generate filename with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.csv"
    
    @staticmethod
    async def save_results(results: List[ProcessingResult], output_base: str):
        df = pd.DataFrame([result.dict() for result in results])
        output_path = ResultsManager.get_timestamped_filename(output_base)
        df.to_csv(output_path, index=False)
        return output_path
    
    @staticmethod
    async def calculate_metrics(results: List[ProcessingResult]) -> Dict:
        """Calculate metrics per category"""
        # Group results by category
        results_by_category = {}
        for result in results:
            if result.category not in results_by_category:
                results_by_category[result.category] = []
            results_by_category[result.category].append(result)
        
        # Calculate metrics for each category
        metrics = {
            "overall": {
                "total_samples": len(results),
                "categories": len(results_by_category)
            },
            "per_category": {}
        }
        
        for category, category_results in results_by_category.items():
            human_labels = [int(r.human_code) for r in category_results]  # Convert to int
            ai_labels = [int(r.ai_code) for r in category_results]  # Convert to int
            total = len(category_results)
            agreements = sum(h == a for h, a in zip(human_labels, ai_labels))
            
            try:
                kappa = cohen_kappa_score(human_labels, ai_labels)
            except ValueError:
                kappa = 0.0
            
            metrics["per_category"][category] = {
                "kappa_score": kappa,
                "total_samples": total,
                "agreement_rate": agreements / total if total > 0 else 0.0,
                "average_confidence": sum(r.confidence for r in category_results) / total
            }
        
        # Calculate overall metrics
        all_human = [int(r.human_code) for r in results]
        all_ai = [int(r.ai_code) for r in results]
        total_agreements = sum(h == a for h, a in zip(all_human, all_ai))
        
        try:
            overall_kappa = cohen_kappa_score(all_human, all_ai)
        except ValueError:
            overall_kappa = 0.0
        
        metrics["overall"].update({
            "kappa_score": overall_kappa,
            "agreement_rate": total_agreements / len(results) if results else 0.0,
            "average_confidence": sum(r.confidence for r in results) / len(results) if results else 0.0
        })
        
        return metrics

# ============================================================================
# AI and Validation Components
# ============================================================================

class GPTClassificationInput(AgentInput):
    """
    Input structure for GPT classification requests
    
    Fields:
    - prompt: Constructed prompt with category details
    - model: GPT model to use (default: gpt-4)
    - temperature: Controls randomness (0.0 for consistent results)
    """
    prompt: str = Field(..., description="Prompt to send to GPT")
    model: str = Field(default="gpt-4", description="GPT model to use")
    temperature: float = Field(default=0.0, description="Temperature setting")

class GPTClassificationOutput(AgentOutput):
    """
    Output structure from GPT classification
    
    Fields:
    - response: Raw JSON response from GPT containing:
      - value: Classification (0 or 1)
      - confidence: Confidence score
      - reasoning: Explanation for classification
    """
    response: str = Field(..., description="Raw GPT response")

class ResponseValidator:
    """
    Validates and interprets GPT classification responses
    
    Responsibilities:
    1. Parse JSON responses from GPT
    2. Validate classification values (0 or 1)
    3. Validate and interpret confidence scores
    4. Handle parsing and validation errors
    """
    
    @staticmethod
    def get_confidence_level(confidence: float) -> ConfidenceLevel:
        """
        Convert numerical confidence to interpretable level
        
        Thresholds:
        - < 0.34: LOW confidence
        - 0.34-0.66: MEDIUM confidence
        - > 0.66: HIGH confidence
        """
        if confidence < 0.34:
            return ConfidenceLevel.LOW
        elif confidence < 0.67:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.HIGH
    
    def validate_response(self, response: str, logger: logging.Logger = None) -> ValidationResult:
        """
        Validate and interpret GPT's classification response
        
        Process:
        1. Parse JSON response
        2. Validate classification value (0 or 1)
        3. Validate confidence score (0.0 to 1.0)
        4. Interpret confidence level
        5. Extract reasoning
        
        Args:
            response: Raw JSON response from GPT
            logger: Optional logger for validation issues
            
        Returns:
            ValidationResult with validated and interpreted values
        """
        try:
            # Parse JSON response
            result = json.loads(response)
            
            # Validate value (must be "0" or "1")
            value = str(result.get("value", "0"))
            if value not in ["0", "1"]:
                raise ValueError(f"Invalid value: {value}. Must be '0' or '1'")
            
            # Validate and interpret confidence
            confidence = float(result.get("confidence", 0.0))
            if not 0 <= confidence <= 1:
                raise ValueError(f"Confidence must be between 0 and 1, got: {confidence}")
            
            confidence_level = self.get_confidence_level(confidence)
            
            # Get reasoning
            reasoning = str(result.get("reasoning", "No reasoning provided"))
            
            if logger:
                logger.debug(f"Validated response with {confidence_level} confidence")
            
            return ValidationResult(
                value=value,
                confidence=confidence,
                confidence_level=confidence_level,
                reasoning=reasoning
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            if logger:
                logger.warning(f"Validation error: Failed to parse response - {str(e)}")
            
            # Return safe defaults on error
            return ValidationResult(
                value="0",
                confidence=0.0,
                confidence_level=ConfidenceLevel.LOW,
                reasoning=f"Error parsing response: {str(e)}"
            )

# GPT agent handles the AI interaction
class GPTClassificationAgent(AtomicAgent):
    """
    AI agent for classifying training data entries
    
    Responsibilities:
    1. Manage OpenAI API connection
    2. Send prompts to GPT
    3. Handle API errors and retries
    4. Return structured responses
    
    Note: Uses AtomicAgent for consistent async processing
    """
    def __init__(self):
        super().__init__()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"))

    async def _process(self, input_data: GPTClassificationInput, context: AgentContext) -> GPTClassificationOutput:
        """
        Process a single classification request
        
        Steps:
        1. Prepare GPT messages (system + user prompt)
        2. Send request to OpenAI API
        3. Extract and return response
        
        Raises:
            AgentError: If API call fails or response is invalid
        """
        try:
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
            return GPTClassificationOutput(response=response.choices[0].message.content)
        except Exception as e:
            raise AgentError(f"Classification error: Failed to get GPT response - {str(e)}")

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
        self.response_validator = ResponseValidator()  # Now a regular validator

    async def process_entry(self, entry: DataEntry, template: str, scheme: CodingScheme) -> List[ProcessingResult]:
        """
        Process a single training entry across all categories
        
        Process:
        1. For each category in scheme:
           a. Generate category-specific prompt
           b. Get GPT classification
           c. Validate and interpret response
           d. Store results
        
        Args:
            entry: Training entry to classify
            template: Base prompt template
            scheme: Validated coding scheme
            
        Returns:
            List of ProcessingResult, one per category
        """
        results = []
        
        for category_key in scheme.categories.keys():
            self.logger.info(f"Processing category: {category_key}")
            
            # Generate prompt and get classification
            prompt = await self.resource_manager.construct_prompt(
                template, entry, scheme, category_key
            )
            gpt_input = GPTClassificationInput(
                prompt=prompt,
                model=self.config['gpt']['model'],
                temperature=self.config['gpt']['temperature']
            )
            gpt_output = await self.classification_agent.process(gpt_input)
            
            # Validate response
            validation_result = self.response_validator.validate_response(
                gpt_output.response,
                logger=self.logger
            )
            
            # Log confidence level
            self.logger.info(
                f"Category {category_key} classified with {validation_result.confidence_level} "
                f"confidence ({validation_result.confidence:.2f})"
            )
            
            # Store result
            results.append(ProcessingResult(
                title=entry.title,
                description=entry.description,
                category=category_key,
                human_code=entry.human_code,
                ai_code=validation_result.value,
                confidence=validation_result.confidence,
                reasoning=validation_result.reasoning
            ))
        
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
            
            # Load scheme with validation
            try:
                scheme = await self.resource_manager.load_scheme(self.config['paths']['coding_scheme'])
                self.logger.info(f"Loaded coding scheme with {len(scheme.categories)} categories")
            except ValueError as e:
                self.logger.error(f"Failed to load coding scheme: {str(e)}")
                raise
            
            template = await self.resource_manager.load_template(self.config['paths']['prompt_template'])
            
            # Process entries with validated scheme
            all_results = []
            entry_count = 0
            async for entry in self.data_manager.iterate_entries(dataset):
                entry_count += 1
                self.logger.info(f"Processing entry {entry_count}: {entry.title}")
                results = await self.process_entry(entry, template, scheme)
                all_results.extend(results)
            
            # Save results with timestamp
            output_path = await self.results_manager.save_results(
                all_results, 
                self.config['paths']['output_base']
            )
            self.logger.info(f"Results saved to: {output_path}")
            
            # Calculate and display metrics
            metrics = await self.results_manager.calculate_metrics(all_results)
            print("\nEvaluation Metrics:")
            print(f"Cohen's Kappa Score: {metrics['overall']['kappa_score']:.2f}")
            print(f"Agreement Rate: {metrics['overall']['agreement_rate']:.2%}")
            print(f"Total Samples: {metrics['overall']['total_samples']}")
            
            self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

async def main():
    """
    Main entry point for the classification pipeline
    
    Process:
    1. Setup environment:
       - Create logs directory
       - Configure logging
       
    2. Initialize and run classifier:
       - Create TrainingDataClassifier with config
       - Run classification process
       - Handle any errors
    """
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging with both console and file output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(CONFIG['logging']['file'])  # File output
        ]
    )
    
    # Run classifier
    classifier = TrainingDataClassifier(CONFIG)
    await classifier.run()

if __name__ == "__main__":
    asyncio.run(main())
