"""
Configuration file for ResearchGPT Assistant

TODO: Complete the following tasks:
1. Set up Mistral API configuration
2. Define file paths for data directories
3. Set up logging configuration
4. Define model parameters (temperature, max_tokens, etc.)
"""

import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        load_dotenv(dotenv_path=env_path)
        
        # Mistral API settings
        self.MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self.MODEL_NAME = "mistral-medium"  # Choose appropriate Mistral model
        self.TEMPERATURE = 0.1  # Set temperature for consistent responses
        self.MAX_TOKENS = 1000  # Set maximum response length
        
        # TODO: Directory paths
        self.DATA_DIR = "data/"
        self.SAMPLE_PAPERS_DIR = "data/sample_papers/"
        self.PROCESSED_DIR = "data/processed/"
        self.RESULTS_DIR = "results/"
        
        # TODO: Processing parameters
        self.CHUNK_SIZE = 1000  # TODO: Set text chunk size for processing
        self.OVERLAP = 100      # TODO: Set overlap between chunks