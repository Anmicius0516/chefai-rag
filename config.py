import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY")
BASE_URL = os.getenv("ZHIPU_BASE_URL")
RERANK_URL = os.getenv("ZHIPU_RERANK_URL")

DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./temp_upload")

# 控制开关
RERANK_ENABLED = True
MEMORY_ENABLED = True
MAX_CONTEXT_DOCS = 6
TOP_K_VECTOR = 4
TOP_K_BM25 = 4