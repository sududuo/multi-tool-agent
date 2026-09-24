import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# 1. 加载环境变量
# ============================================================

load_dotenv()


# ============================================================
# 2. 项目路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

COMPANY_FILE = (
    BASE_DIR
    / "data"
    / "company.txt"
)

DB_PATH = (
    BASE_DIR
    / "chroma_db"
)

CHECKPOINT_PATH = (
    BASE_DIR
    / "checkpoints.sqlite"
)


# ============================================================
# 3. API 配置
# ============================================================

DATA_BASE_URL = os.getenv(
    "DATA_BASE_URL"
)

DATA_MODEL = os.getenv(
    "DATA_MODEL"
)
DASHSCOPE_API_KEY=os.getenv(
    "DASHSCOPE_API_KEY"
)
DASHSCOPE_BASE_URL=os.getenv(
    "DASHSCOPE_BASE_URL"
)



# ============================================================
# 4. Token 配置
# ============================================================

SUMMARY_TOKEN_THRESHOLD = 4000

SUMMARY_INTERVAL_TOKEN = 2000

CONTEXT_TOKEN_LIMIT = 2500