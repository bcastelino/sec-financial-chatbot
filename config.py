import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenRouter API Configuration
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    # Model Configuration
    DEFAULT_MODEL = "deepseek/deepseek-r1-0528:free"

    # Companies to collect data for
    TOP_COMPANIES = [
        {"ticker": "AAPL", "name": "Apple Inc.", "aliases": ["apple"], "icon": "🎧", "sector": "Consumer Electronics"},
        {"ticker": "MSFT", "name": "Microsoft Corporation", "aliases": ["microsoft"], "icon": "💻", "sector": "Software & Cloud Services"},
        {"ticker": "GOOGL", "name": "Alphabet Inc.", "aliases": ["google", "alphabet"], "icon": "🌍", "sector": "Digital Advertising & Cloud"},
        {"ticker": "AMZN", "name": "Amazon.com Inc.", "aliases": ["amazon"], "icon": "🛒", "sector": "E-commerce & Cloud Services"},
        {"ticker": "NVDA", "name": "NVIDIA Corporation", "aliases": ["nvidia"], "icon": "🎮", "sector": "Semiconductors & AI"},
        {"ticker": "META", "name": "Meta Platforms Inc.", "aliases": ["meta", "facebook"], "icon": "📱", "sector": "Social Media & Digital Advertising"}
    ]

    # SEC EDGAR identity (for user-agent header)
    EDGAR_IDENTITY = os.getenv("EDGAR_IDENTITY", "Your Name your.email@example.com")

    # Year range for filings
    START_YEAR = 2020
    END_YEAR = 2025

    # Vector Database Configuration
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"  # Path to ChromaDB storage
    COLLECTION_NAME = "sec_filings"           # ChromaDB collection name

    # Embedding Model Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"      # Sentence transformer model for embeddings

    # RAG/Chunking Configuration
    CHUNK_SIZE = 1000         # Legacy parameter (kept for backward compatibility)
    MAX_CHUNK_SIZE = 3000     # Maximum characters per text chunk (new parameter)
    CHUNK_OVERLAP = 300       # Overlap between text chunks (updated default)
    DEFAULT_NUM_TEXT_CHUNKS = 50  # Number of text chunks to retrieve for LLM context

    # File paths
    DATA_DIR = "./data"
    FILINGS_DIR = "./data/filings"
    XBRL_DIR = "./data/xbrl"

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [cls.DATA_DIR, cls.FILINGS_DIR, cls.XBRL_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True) 