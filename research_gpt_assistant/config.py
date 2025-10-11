# config.py
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
        
        
        # Directory paths
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        self.SAMPLE_PAPERS_DIR = os.path.join(self.DATA_DIR, "sample_papers")  # <-- point to PDFs
        self.PROCESSED_DIR = os.path.join(self.DATA_DIR, "processed")
        self.RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
        
        # TODO: Processing parameters
        self.CHUNK_SIZE = 1000  # TODO: Set text chunk size for processing
        self.OVERLAP = 100      # TODO: Set overlap between chunks
