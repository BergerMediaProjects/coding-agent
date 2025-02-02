"""
AI Training Data Pipeline

This pipeline processes training data through a combination of:
1. Service Components: Handle basic operations like loading and saving data
2. AI Agents: Make decisions about categorization and handle complex processing

The pipeline maintains clean separation between AI and non-AI tasks for clarity and maintainability.
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

# Configuration
CONFIG = {
    'paths': {
        'data_csv': "teacher_training_data.csv",
        'human_codes': "human_codes.csv",
        'coding_scheme': "old_data_training/coding_scheme.yml",
        'prompt_template': "prompt.txt",
        'output_base': "ai_coded_results"
    },
    'logging': {
        'level': 'INFO',
        'file': os.path.join('logs', 'atomic_agents.log')
    },
    'gpt': {
        'model': 'gpt-4',
        'temperature': 0.0
    }
}

# ============================================================================
# Data Structures
# ============================================================================

class DataEntry(BaseModel):
    """Structure for a single data entry"""
    title: str
    description: str
    human_code: str = Field(
        default="0",
        pattern="^[01]$",
        description="Human-assigned code (0 or 1)"
    )

class ProcessingResult(BaseModel):
    """Structure for processing results"""
    title: str
    description: str
    category: str = Field(..., description="Category being evaluated")
    human_code: str = Field(..., pattern="^[01]$", description="Human-assigned code (0 or 1)")
    ai_code: str = Field(..., pattern="^[01]$", description="AI-assigned code (0 or 1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="AI confidence score")
    reasoning: str = Field(..., description="AI reasoning for the classification")

# ============================================================================
# Service Components
# ============================================================================

class DataManager:
    """Manages training data loading and processing"""
    @staticmethod
    async def load_data(path: str) -> pd.DataFrame:
        return pd.read_csv(path)
    
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
    """Manages loading and preparation of classification resources"""
    @staticmethod
    async def load_scheme(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    
    @staticmethod
    async def load_template(path: str) -> str:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    
    @staticmethod
    async def construct_prompt(template: str, entry: DataEntry, scheme: dict, category_key: str) -> str:
        """Constructs prompt for a specific category from the coding scheme"""
        # Get the category details from scheme
        category_data = scheme[category_key]
        
        # Replace basic placeholders
        prompt = template.replace("[title]", entry.title)
        prompt = prompt.replace("[description]", entry.description)
        
        # Replace scheme-specific placeholders
        prompt = prompt.replace("[category_name]", category_key)
        prompt = prompt.replace("[criteria]", category_data['criteria'])
        prompt = prompt.replace("[values]", category_data['values'])
        
        # Format examples list with proper indentation
        examples_text = "\n".join(f"- {example}" for example in category_data['examples'])
        prompt = prompt.replace("[examples]", examples_text)
        
        # Add category validation hint
        prompt = prompt.replace('"[category_name]"', f'"{category_key}"')
        
        return prompt

class ResultsManager:
    """Manages classification results and metrics"""
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
# AI Agents
# ============================================================================

class GPTClassificationInput(AgentInput):
    """Input schema for GPT classification"""
    prompt: str = Field(..., description="Prompt to send to GPT")
    model: str = Field(default="gpt-4", description="GPT model to use")
    temperature: float = Field(default=0.0, description="Temperature setting")

class GPTClassificationOutput(AgentOutput):
    """Output schema for GPT classification"""
    response: str = Field(..., description="Raw GPT response")

class ResponseInterpreterInput(AgentInput):
    """Input schema for response interpretation"""
    response: str = Field(..., description="Raw GPT response to interpret")

class ResponseInterpreterOutput(AgentOutput):
    """Output schema for response interpretation"""
    value: str = Field(
        ..., 
        description="Classification value (0 or 1)",
        pattern="^[01]$"  # Ensure only "0" or "1" are allowed
    )
    confidence: float = Field(
        ..., 
        description="Confidence score",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(..., description="Reasoning for the classification")

class GPTClassificationAgent(AtomicAgent):
    """AI agent for classifying training data entries"""
    def __init__(self):
        super().__init__()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"))

    async def _process(self, input_data: GPTClassificationInput, context: AgentContext) -> GPTClassificationOutput:
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
            raise AgentError(f"GPT classification error: {str(e)}")

class ResponseInterpreter(AtomicAgent):
    """AI agent for interpreting classification responses"""
    async def _process(self, input_data: ResponseInterpreterInput, context: AgentContext) -> ResponseInterpreterOutput:
        try:
            result = json.loads(input_data.response)
            
            # Validate and extract fields
            value = str(result.get("value", "0"))
            if value not in ["0", "1"]:
                raise ValueError(f"Invalid value: {value}. Must be '0' or '1'")
            
            confidence = float(result.get("confidence", 0.0))
            if not 0 <= confidence <= 1:
                raise ValueError(f"Confidence must be between 0 and 1, got: {confidence}")
            
            reasoning = str(result.get("reasoning", "No reasoning provided"))
            
            return ResponseInterpreterOutput(
                value=value,
                confidence=confidence,
                reasoning=reasoning
            )
        except (json.JSONDecodeError, ValueError) as e:
            context.metadata["interpretation_error"] = str(e)
            return ResponseInterpreterOutput(
                value="0",
                confidence=0.0,
                reasoning=f"Error parsing response: {str(e)}"
            )

# ============================================================================
# Pipeline Coordinator
# ============================================================================

class TrainingDataClassifier:
    """Coordinates the classification of training data using AI agents and services"""
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger("training_classifier")
        
        # Initialize managers
        self.data_manager = DataManager()
        self.resource_manager = ResourceManager()
        self.results_manager = ResultsManager()
        
        # Initialize AI agents
        self.classification_agent = GPTClassificationAgent()
        self.response_interpreter = ResponseInterpreter()

    async def process_entry(self, entry: DataEntry, template: str, scheme: dict) -> List[ProcessingResult]:
        """Process a single entry for all categories in the scheme"""
        context = AgentContext()
        results = []
        
        # Process each category
        for category_key in scheme.keys():
            self.logger.info(f"Processing category: {category_key}")
            
            # Construct category-specific prompt
            prompt = await self.resource_manager.construct_prompt(
                template, 
                entry, 
                scheme,
                category_key
            )
            
            # Get GPT classification using config
            gpt_input = GPTClassificationInput(
                prompt=prompt,
                model=self.config['gpt']['model'],
                temperature=self.config['gpt']['temperature']
            )
            gpt_output = await self.classification_agent.process(gpt_input, context)
            
            # Parse response
            interpreter_input = ResponseInterpreterInput(response=gpt_output.response)
            interpreter_output = await self.response_interpreter.process(interpreter_input, context)
            
            # Check for errors
            if "interpretation_error" in context.metadata:
                self.logger.warning(
                    f"Interpretation error for category {category_key}: "
                    f"{context.metadata['interpretation_error']}"
                )
            
            # Store result
            results.append(ProcessingResult(
                title=entry.title,
                description=entry.description,
                category=category_key,
                human_code=entry.human_code,
                ai_code=interpreter_output.value,  # Changed from category to value
                confidence=interpreter_output.confidence,
                reasoning=interpreter_output.reasoning
            ))
            
        return results

    async def run(self):
        """Run the complete classification pipeline"""
        self.logger.info("Starting classification")
        try:
            # Load resources
            dataset = await self.data_manager.load_data(self.config['paths']['data_csv'])
            if self.config['paths'].get('human_codes'):
                codes = await self.data_manager.load_data(self.config['paths']['human_codes'])
                dataset = await self.data_manager.merge_datasets(dataset, codes)
            
            scheme = await self.resource_manager.load_scheme(self.config['paths']['coding_scheme'])
            template = await self.resource_manager.load_template(self.config['paths']['prompt_template'])
            
            # Process entries
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
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(CONFIG['logging']['file'])
        ]
    )
    
    # Run pipeline
    pipeline = TrainingDataClassifier(CONFIG)
    await pipeline.run()

if __name__ == "__main__":
    asyncio.run(main())
