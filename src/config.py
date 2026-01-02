import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    PDF_DATA_PATH = Path(os.getenv("PDF_DATA_PATH", "./data/pdfs"))
    PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "./data/processed"))
    CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
    
    # Embeddings
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
    
    # LLM Settings
    MAX_TOKENS = 4096
    TEMPERATURE = 0.1
    
    # Extraction Patterns
    ACTIVITY_PATTERNS = [
        "drilling", "drill", "drilled",
        "reaming", "ream", "reamed",
        "trip", "tripping", "pooh", "rih",
        "circulating", "circulation",
        "casing", "cementing",
        "bop", "test", "testing",
        "fishing", "stuck pipe",
        "lost circulation",
        "survey", "surveying"
    ]
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories"""
        cls.PDF_DATA_PATH.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Ensure directories exist
Config.ensure_dirs()