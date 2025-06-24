from flask import Flask, render_template, request, jsonify, send_file, Response, session
import os
import shutil
import asyncio
import sys
import yaml
import glob
import time
import uuid
from datetime import datetime
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import json
from utils.yaml_generator import YAMLGenerator
from utils.fix_yaml_format import fix_yaml_format
from utils.validate_yaml import validate_yaml
import signal
import logging
from logging.handlers import RotatingFileHandler
import traceback
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

# Get the root directory path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Add the parent directory to the Python path
sys.path.append(root_dir)
from pipeline_draft1_0 import TrainingDataClassifier, CONFIG

# Verify API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Configure logging
def setup_logging():
    """Setup logging configuration for the web interface"""
    # Create log directory if it doesn't exist
    log_dir = os.path.join(root_dir, 'data', 'log')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create run-specific log file with rotation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'web_interface_{timestamp}.log')
    
    # Setup logging with custom formatting and rotation
    formatter = logging.Formatter(
        json.dumps({
            'timestamp': '%(asctime)s',
            'level': '%(levelname)s',
            'logger': '%(name)s',
            'message': '%(message)s',
            'module': '%(module)s',
            'function': '%(funcName)s',
            'line': '%(lineno)d'
        }),
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger('web_interface')

# Initialize logger
logger = setup_logging()

# Debug: Print environment variables
logger.info("Flask Environment Variables", extra={
    'current_directory': os.getcwd(),
    'root_directory': root_dir,
    'openai_api_key_exists': 'OPENAI_API_KEY' in os.environ,
    'openai_api_key_length': len(api_key) if 'OPENAI_API_KEY' in os.environ else 0,
    'openai_api_key_prefix': api_key[:8] + '...' if 'OPENAI_API_KEY' in os.environ else None
})

# Configure upload folders
DEFAULT_FOLDER = os.path.join(root_dir, 'data', 'DOC_coding_scheme')
TEMP_FOLDER = os.path.join(root_dir, 'data', 'temp_uploads')
ALLOWED_EXTENSIONS = {
    'coding_scheme': {'yml', 'yaml'},
    'prompt': {'txt'},
    'input_data': {'xlsx'}
}

# Create temp directory if it doesn't exist
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Initialize Flask app with async support
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = DEFAULT_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = os.urandom(24)  # Required for session handling

# Load configuration
CONFIG = {
    'paths': {
        'data_csv': os.path.join(root_dir, 'data', 'training_data.xlsx'),
        'human_codes': os.path.join(root_dir, 'data', 'human_codes.xlsx'),
        'coding_scheme': os.path.join(root_dir, 'data', 'DOC_coding_scheme', 'coding_scheme_imported.yml'),
        'prompt_template': os.path.join(root_dir, 'data', 'prompt.txt'),
        'output_base': os.path.join(root_dir, 'data', 'results'),
        'log_dir': os.path.join(root_dir, 'data', 'log')
    },
    'gpt': {
        'model': 'gpt-4-turbo-preview',
        'temperature': 0.0
    },
    'selected_categories': [],
    'temp_files': {
        'data_csv': None,
        'coding_scheme': None,
        'prompt_template': None
    }
}

# Add status tracking
pipeline_status = {
    'is_running': False,
    'current_entry': 0,
    'total_entries': 0,
    'current_category': '',
    'progress': 0,
    'status_message': '',
    'error': None,
    'is_cancelled': False  # Add cancellation flag
}

def get_session_folder():
    """Get or create a session-specific folder for uploads"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    session_folder = os.path.join(TEMP_FOLDER, session['session_id'])
    os.makedirs(session_folder, exist_ok=True)
    return session_folder

def cleanup_old_sessions():
    """Clean up sessions older than 24 hours"""
    now = time.time()
    for session_folder in os.listdir(TEMP_FOLDER):
        folder_path = os.path.join(TEMP_FOLDER, session_folder)
        if os.path.isdir(folder_path):
            # Get the last modification time
            mtime = os.path.getmtime(folder_path)
            # If older than 24 hours, delete it
            if now - mtime > 86400:  # 24 hours in seconds
                shutil.rmtree(folder_path)

def allowed_file(filename, file_type):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS.get(file_type, set())

def get_latest_results_file():
    """Get the path to the most recent results file."""
    results_dir = os.path.join(root_dir, 'data', 'results')
    # Look for both naming patterns
    files = glob.glob(os.path.join(results_dir, 'ai_coded_results_*.xlsx')) + \
            glob.glob(os.path.join(results_dir, 'results_*.xlsx'))
    if not files:
        return None
    return max(files, key=os.path.getctime)

def get_category_sort_key(display_name):
    """
    Convert category name like '3.5' or '3.1.2' or '2.0b' or '2.0.1' into a tuple of numbers for proper sorting
    Example: '3.5' -> (3, 5, 0)
            '3.1.2' -> (3, 1, 2)
            '3.1' -> (3, 1, 0)
            '2.0b' -> (2, 0, 0, 'b')  # Letter suffix preserved for sorting
            '2.0.1' -> (2, 0, 1)  # Handle subcategories
    """
    # Extract the numeric part at the start of the display name
    parts = display_name.split()
    if not parts:
        return (999, 999, 999, '')  # Handle empty strings
    
    # Get the first part which should be the number (e.g., "2.6.5" or "3.5" or "2.0b" or "2.0.1")
    number_part = parts[0]
    
    # Split by dots
    number_sections = number_part.split('.')
    
    # Initialize numbers list and letter suffix
    numbers = []
    letter_suffix = ''
    
    try:
        # Process each section
        for section in number_sections:
            # Check if section contains letters
            numeric_part = ''.join(c for c in section if c.isdigit())
            letter_part = ''.join(c for c in section if c.isalpha())
            
            if numeric_part:
                numbers.append(int(numeric_part))
            if letter_part:
                letter_suffix = letter_part
                
        # Pad with zeros if needed
        while len(numbers) < 3:
            numbers.append(0)
            
        # Return tuple with numbers and letter suffix
        return tuple(numbers[:3]) + (letter_suffix,)
    except Exception:
        return (999, 999, 999, '')

def filter_categories(scheme):
    """Filter categories based on specific criteria"""
    filtered_scheme = {}
    categories_with_sort_keys = []
    
    # Get categories from the new nested structure
    categories = scheme.get('coding_scheme', {}).get('categories', {})
    
    for category, details in categories.items():
        # Get the display name (or use the category key if display_name is not present)
        display_name = details.get('display_name', category)
        
        # Extract the numeric prefix from display_name
        # Example: "2.0.1 Vorkommen Medienkompetenz" -> "2.0.1"
        numeric_prefix = display_name.split()[0] if display_name else ""
        
        # Check if category starts with "2." or "3." and doesn't start with "_DERIVED_"
        if (numeric_prefix.startswith(('2.', '3.')) and 
            not category.startswith('_DERIVED_') and
            not numeric_prefix.startswith('1.')):
            # Store category with its sort key
            sort_key = get_category_sort_key(numeric_prefix)
            categories_with_sort_keys.append((sort_key, category, details))
    
    # Sort categories based on their numeric parts and letter suffixes
    categories_with_sort_keys.sort(key=lambda x: x[0])
    
    # Create ordered dictionary
    for _, category, details in categories_with_sort_keys:
        filtered_scheme[category] = details
    
    return filtered_scheme

@app.route('/')
def index():
    # Load current coding scheme to get categories
    coding_scheme_path = CONFIG['paths']['coding_scheme']
    with open(coding_scheme_path, 'r', encoding='utf-8') as file:
        scheme = yaml.safe_load(file)
    
    # Filter categories
    filtered_scheme = filter_categories(scheme)
    return render_template('index.html', categories=filtered_scheme)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    file_type = request.form.get('type')
    
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if not file_type or file_type not in ALLOWED_EXTENSIONS:
        logger.error("Invalid file type", extra={'file_type': file_type})
        return jsonify({'error': 'Invalid file type'}), 400
    
    if not allowed_file(file.filename, file_type):
        logger.error("File type not allowed", extra={
            'filename': file.filename,
            'file_type': file_type
        })
        return jsonify({'error': f'File type not allowed for {file_type}'}), 400
    
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    session_folder = os.path.join(TEMP_FOLDER, session_id)
    os.makedirs(session_folder, exist_ok=True)
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(session_folder, filename)
    file.save(file_path)
    
    logger.info("File uploaded successfully", extra={
        'filename': filename,
        'file_type': file_type,
        'session_id': session_id,
        'file_path': file_path
    })
    
    # Update configuration based on file type
    if file_type == 'coding_scheme':
        CONFIG['paths']['coding_scheme'] = file_path
    elif file_type == 'prompt':
        CONFIG['paths']['prompt_template'] = file_path
    elif file_type == 'input_data':
        CONFIG['paths']['data_csv'] = file_path
    
    return jsonify({'message': 'File uploaded successfully'})

@app.route('/pipeline_status', methods=['GET'])
def get_pipeline_status():
    """Get current pipeline status"""
    return jsonify(pipeline_status)

@app.route('/cancel_pipeline', methods=['POST'])
def cancel_pipeline():
    """Cancel the running pipeline"""
    global pipeline_status
    if pipeline_status['is_running']:
        pipeline_status['is_cancelled'] = True
        pipeline_status['status_message'] = 'Cancellation requested...'
        return jsonify({'status': 'success', 'message': 'Pipeline cancellation requested'})
    return jsonify({'status': 'error', 'message': 'No pipeline running'}), 400

@app.route('/run_pipeline', methods=['POST'])
async def run_pipeline():
    global pipeline_status
    try:
        # Reset status
        pipeline_status = {
            'is_running': True,
            'current_entry': 0,
            'total_entries': 0,
            'current_category': '',
            'progress': 0,
            'status_message': 'Initializing pipeline...',
            'error': None,
            'is_cancelled': False
        }
        
        # Get uploaded files
        data_file = request.files.get('data_file')
        coding_scheme_file = request.files.get('coding_scheme_file')
        prompt_file = request.files.get('prompt_file')
        
        # Get selected categories
        selected_categories = request.form.get('selected_categories')
        if selected_categories:
            try:
                selected_categories = json.loads(selected_categories)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing selected categories: {e}")
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid categories format'
                }), 400
        else:
            selected_categories = []

        logger.info("Starting pipeline run", extra={
            'has_data_file': bool(data_file),
            'has_coding_scheme': bool(coding_scheme_file),
            'has_prompt_file': bool(prompt_file),
            'selected_categories': selected_categories
        })

        # Get session folder for temporary files
        session_folder = get_session_folder()

        # Save temporary files if provided
        temp_files = {}
        if data_file:
            pipeline_status['status_message'] = 'Saving data file...'
            data_path = os.path.join(session_folder, 'temp_data.xlsx')
            data_file.save(data_path)
            temp_files['data_csv'] = data_path
            logger.info("Saved data file", extra={'path': data_path})
            
        if coding_scheme_file:
            pipeline_status['status_message'] = 'Processing coding scheme...'
            # Save DOCX file
            docx_path = os.path.join(session_folder, 'temp_coding_scheme.docx')
            coding_scheme_file.save(docx_path)
            temp_files['docx_file'] = docx_path
            logger.info("Saved coding scheme file", extra={'path': docx_path})
            
            # Generate YAML from DOCX
            yaml_generator = YAMLGenerator()
            yaml_path = os.path.join(session_folder, 'temp_coding_scheme.yml')
            if not yaml_generator.generate_yaml_from_docx(docx_path, yaml_path):
                pipeline_status['error'] = 'Failed to generate YAML from DOCX'
                pipeline_status['is_running'] = False
                logger.error("Failed to generate YAML from DOCX")
                return jsonify({'status': 'error', 'message': 'Failed to generate YAML from DOCX'}), 500
            
            # Fix YAML format
            if not fix_yaml_format(yaml_path, yaml_path):
                pipeline_status['error'] = 'Failed to fix YAML format'
                pipeline_status['is_running'] = False
                logger.error("Failed to fix YAML format")
                return jsonify({'status': 'error', 'message': 'Failed to fix YAML format'}), 500
            
            # Validate YAML
            if not validate_yaml(yaml_path):
                pipeline_status['error'] = 'Invalid YAML format'
                pipeline_status['is_running'] = False
                logger.error("Invalid YAML format")
                return jsonify({'status': 'error', 'message': 'Invalid YAML format'}), 500
            
            temp_files['coding_scheme'] = yaml_path
            logger.info("Generated and validated YAML", extra={'path': yaml_path})
            
        if prompt_file:
            pipeline_status['status_message'] = 'Saving prompt file...'
            prompt_path = os.path.join(session_folder, 'temp_prompt.txt')
            prompt_file.save(prompt_path)
            temp_files['prompt_template'] = prompt_path
            logger.info("Saved prompt file", extra={'path': prompt_path})

        # Update CONFIG with temporary files
        for key, path in temp_files.items():
            if key in CONFIG['paths']:
                CONFIG['paths'][key] = path

        # Update selected categories
        CONFIG['selected_categories'] = selected_categories

        # Run the pipeline
        pipeline_status['status_message'] = 'Starting classification...'
        classifier = TrainingDataClassifier(CONFIG)
        
        # Set up status update callback
        def status_callback(entry_num, total_entries, category, progress):
            pipeline_status.update({
                'current_entry': entry_num,
                'total_entries': total_entries,
                'current_category': category,
                'progress': progress,
                'status_message': f'Processing entry {entry_num}/{total_entries} - Category: {category}',
                'is_running': True  # Ensure running state is maintained
            })
            
            # Check for cancellation
            if pipeline_status['is_cancelled']:
                raise Exception("Pipeline cancelled by user")
        
        # Add callback to CONFIG
        CONFIG['status_callback'] = status_callback
        
        try:
            result = await classifier.run()
        except Exception as e:
            if pipeline_status['is_cancelled']:
                pipeline_status.update({
                    'is_running': False,
                    'status_message': 'Pipeline cancelled by user',
                    'error': 'Pipeline cancelled by user'
                })
                return jsonify({
                    'status': 'cancelled',
                    'message': 'Pipeline cancelled by user'
                })
            raise

        # Get the latest results file
        result_file = get_latest_results_file()
        if result_file:
            result_filename = os.path.basename(result_file)
            pipeline_status.update({
                'is_running': False,
                'status_message': 'Pipeline completed successfully',
                'progress': 100
            })
            logger.info("Pipeline completed successfully", extra={
                'result_file': result_filename
            })
            return jsonify({
                'status': 'success',
                'message': 'Pipeline completed successfully',
                'result_file': result_filename
            })
        else:
            pipeline_status.update({
                'is_running': False,
                'error': 'No results file found',
                'status_message': 'Pipeline completed but no results file was found'
            })
            logger.error("Pipeline completed but no results file was found")
            return jsonify({
                'status': 'error',
                'message': 'Pipeline completed but no results file was found'
            }), 500

    except Exception as e:
        error_message = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"Error running pipeline: {error_message}\n{stack_trace}")
        
        pipeline_status.update({
            'is_running': False,
            'error': error_message,
            'status_message': f'Error running pipeline: {error_message}'
        })
        
        return jsonify({
            'status': 'error',
            'message': f'Error running pipeline: {error_message}'
        }), 500

@app.route('/download_results/<filename>')
def download_results(filename):
    results_dir = os.path.join(root_dir, 'data', 'results')
    file_path = os.path.join(results_dir, filename)
    return send_file(file_path, as_attachment=True)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\nShutting down server...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Add error handlers
@app.errorhandler(Exception)
def handle_error(error):
    """Handle all unhandled exceptions"""
    error_message = str(error)
    stack_trace = traceback.format_exc()
    logger.error(f"Unhandled error: {error_message}\n{stack_trace}")
    
    # Update pipeline status if it exists
    global pipeline_status
    if 'pipeline_status' in globals():
        pipeline_status.update({
            'is_running': False,
            'error': error_message,
            'status_message': f'Error: {error_message}'
        })
    
    return jsonify({
        'status': 'error',
        'message': error_message
    }), 500

@app.errorhandler(404)
def handle_404(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Resource not found'
    }), 404

@app.errorhandler(405)
def handle_405(error):
    """Handle 405 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Method not allowed'
    }), 405

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='127.0.0.1') 
