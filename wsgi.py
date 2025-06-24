import sys
import os

# Add your project directory to the Python path
project_home = os.path.dirname(os.path.abspath(__file__))
if project_home not in sys.path:
    sys.path.append(project_home)

# Load environment variables
from dotenv import load_dotenv
env_path = os.path.join(project_home, '.env')
load_dotenv(env_path, override=True)  # Use override=True to ensure the values are set

# Verify API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Debug: Print environment variables
print("\n=== WSGI Environment Variables ===")
print(f"Project home: {project_home}")
print(f"Env file path: {env_path}")
print(f"OPENAI_API_KEY exists: {'OPENAI_API_KEY' in os.environ}")
if 'OPENAI_API_KEY' in os.environ:
    api_key = os.environ['OPENAI_API_KEY']
    print(f"OPENAI_API_KEY length: {len(api_key)}")
    print(f"OPENAI_API_KEY first 8 chars: {api_key[:8]}...")
print("==========================\n")

# Import your Flask app
from web_interface.app import app 