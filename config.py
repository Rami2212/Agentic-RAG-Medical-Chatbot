import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

# Data Configuration
QNA_DATA_PATH = "data/medical_qna_dataset.csv"
DEVICE_DATA_PATH = "data/medical_device_manuals_dataset.csv"
SAMPLE_SIZE = 500
RANDOM_STATE = 42

# ChromaDB Configuration
CHROMA_DB_PATH = "./chroma_db"
QNA_COLLECTION_NAME = "medical_qna"
DEVICE_COLLECTION_NAME = "medical_device_manuals"

# Retrieval Configuration
TOP_K_RESULTS = 3
MAX_ITERATIONS = 3
ANSWER_LENGTH_LIMIT = 50

# Web Search Configuration
TAVILY_TOPIC = "general"
TAVILY_MAX_RESULTS = 1

