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
        # TODO: Load environment variables
        load_dotenv()
        
        # TODO: Mistral API settings
        self.MISTRAL_API_KEY = "MISTRAL_API_KEY"  # Replace with actual key
        self.MODEL_NAME = "mistral-medium"  # TODO: Choose appropriate Mistral model
        self.TEMPERATURE = 0.1  # TODO: Set temperature for consistent responses
        self.MAX_TOKENS = 1000  # TODO: Set maximum response length
        
        # TODO: Directory paths
        self.DATA_DIR = "data/"
        self.SAMPLE_PAPERS_DIR = "data/sample_papers/"
        self.PROCESSED_DIR = "data/processed/"
        self.RESULTS_DIR = "results/"
        
        # TODO: Processing parameters
        self.CHUNK_SIZE = 1000  # TODO: Set text chunk size for processing
        self.OVERLAP = 100      # TODO: Set overlap between chunks