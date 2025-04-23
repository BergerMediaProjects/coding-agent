# AI Pipeline for Course Description Analysis

This pipeline analyzes course descriptions using AI to classify them according to specific digital competence categories.

## What Can You Do With This Pipeline?

This pipeline helps researchers and educators analyze course descriptions to identify digital competencies. Here's what you can do:

1. **Automated Course Analysis**
   - Analyze large numbers of course descriptions automatically
   - Classify courses according to digital competence categories
   - Get consistent and objective classifications
   - Save time compared to manual coding

2. **Flexible Category System**
   - Use predefined digital competence categories
   - Create your own categories using the YAML generator
   - Modify existing categories easily
   - Activate/deactivate categories as needed

3. **Quality Control**
   - Get confidence scores for each classification
   - Review AI reasoning for each decision
   - Validate YAML structure automatically
   - Cross-check results with manual coding

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## File Structure

```
.
├── README.md
├── requirements.txt
├── pipeline-draft1.0.py      # Main pipeline script
├── coding_scheme.yml        # Active coding scheme (used by pipeline)
├── training_data.xlsx      # Course descriptions
├── human_codes.xlsx        # Optional human-coded data
├── utils/
│   ├── validate_yaml.py      # YAML validation script
│   ├── yaml_generator.py     # Converts Word docs to YAML
│   ├── fix_yaml_format.py    # Cleans up YAML format
│   └── prompt.txt           # GPT prompt template
├── DOC_coding_scheme/       # Directory for coding scheme documents
│   ├── doc_cs.docx         # Word document with coding scheme
│   └── coding_scheme_imported.yml  # Generated YAML scheme
├── log/                    # Log files directory
└── results/                # Results will be saved here
    └── ai_coded_results_*.xlsx   # Timestamped results
```

## Workflow

1. **Generate YAML from Word document**
   ```bash
   python utils/yaml_generator.py
   ```
   This creates `DOC_coding_scheme/coding_scheme_imported.yml`

2. **Fix YAML formatting**
   ```bash
   python utils/fix_yaml_format.py
   ```
   This creates `coding_scheme.yml` in the root directory

3. **Validate YAML structure**
   ```bash
   python utils/validate_yaml.py
   ```
   This checks the root `coding_scheme.yml`

4. **Run the pipeline**
   ```bash
   python pipeline-draft1.0.py
   ```
   This uses the root `coding_scheme.yml`

## Available Scripts

### 1. YAML Generator (`utils/yaml_generator.py`)

Converts the Word document coding scheme to YAML format.

**Usage:**
```bash
python utils/yaml_generator.py
```

**Input:**
- Reads from: `DOC_coding_scheme/doc_cs.docx`
- Expects a table with columns for:
  - Category names
  - Values (including conditions)
  - Criteria descriptions
  - Examples

**Output:**
- Creates: `DOC_coding_scheme/coding_scheme_imported.yml`
- Copy this file to `data/coding_scheme.yml` to use it in the pipeline

### 2. YAML Format Fixer (`utils/fix_yaml_format.py`)

Cleans and standardizes the YAML format after generation.

**Usage:**
```bash
python utils/fix_yaml_format.py
```

**Features:**
- Simplifies category names by removing numbering patterns
- Standardizes formatting of criteria, examples, and values
- Fixes quote consistency and special characters
- Ensures UTF-8 encoding

**Input:**
- Reads from: `DOC_coding_scheme/coding_scheme_imported.yml`

**Output:**
- Creates: `coding_scheme.yml` in the root directory
- Performs these transformations:
  - Removes numbering (e.g., "2.1.1 Category" → "Category")
  - Standardizes quotes and line breaks
  - Fixes bullet points in examples
  - Ensures consistent value formats

**Example Transformations:**
```yaml
# Before:
2.0 a Vorkommen Medienkompetenz:
  criteria: "Kriterium: Some text"
  examples:
    - "• Example 1"
    - "· Example 2"
  values: 'Ja (1), Nein (0)'

# After:
Vorkommen Medienkompetenz:
  criteria: "Some text"
  examples:
    - "Example 1"
    - "Example 2"
  values: "Ja (1), Nein (0)"
```

### 3. YAML Validator (`utils/validate_yaml.py`)

Validates the coding scheme structure.

**Usage:**
```bash
python utils/validate_yaml.py
```

**Default Paths:**
- If no path is provided, it looks for `coding_scheme.yml` in the root directory
- This is the same file that `fix_yaml_format.py` creates as output

### 4. Main Pipeline (`pipeline-draft1.0.py`)

The primary script for analyzing course descriptions.

**Usage:**
```bash
python pipeline-draft1.0.py
```

**Input Requirements:**
- Excel files in root directory:
  - `training_data.xlsx`: Course descriptions (title and description columns)
  - `human_codes.xlsx`: Optional human-coded data
- Valid `coding_scheme.yml` in the root directory
- OpenAI API key in environment

**Configuration:**
- Edit `SELECTED_CATEGORIES` in the script to choose which categories to analyze
- Categories can be activated/deactivated by uncommenting/commenting them

**Output:**
- Creates Excel files in the `results/` directory
- Files are timestamped (format: `ai_coded_results_YYYYMMDD_HHMMSS.xlsx`)
- Results include:
  - Original course titles and descriptions
  - AI classifications (1/0 for binary categories)
  - Confidence scores
  - AI reasoning for each classification

## Value Transformations

The pipeline automatically transforms certain values in the output:
- "Ja", "Ja (1)", "(1)" → "1"
- "Nein", "Nein (0)", "(0)" → "0"

## Best Practices

1. **Input Data**
   - Ensure clean, consistent formatting in input files
   - Remove any special characters from titles/descriptions
   - Use UTF-8 encoding for all files

2. **Category Selection**
   - Start with a few categories for testing
   - Gradually add more categories as needed
   - Check confidence scores to validate results

3. **Output Validation**
   - Review AI reasoning for unexpected classifications
   - Monitor confidence scores for reliability
   - Cross-validate results with manual coding if possible

## Troubleshooting

Common issues and solutions:

1. **File Not Found Errors**
   - Check if all required directories exist
   - Ensure input files are in the correct locations
   - Verify file permissions

2. **YAML Validation Errors**
   - Run the YAML validator
   - Check for proper indentation
   - Verify all required fields are present

3. **Classification Issues**
   - Review the prompt.txt file
   - Adjust category criteria if needed
   - Add more specific examples to the coding scheme

## Updates and Maintenance

To update the pipeline:
1. Pull latest changes from repository
2. Check for new dependencies in requirements.txt
3. Review any changes to the coding scheme
4. Test with a small sample before full analysis

## Contact

For support or questions, please contact sonja.berger@lmu.de.
