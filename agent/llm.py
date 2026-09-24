from langchain_openai import ChatOpenAI

from config import (
    DASHSCOPE_API_KEY,
    DATA_BASE_URL,
    DATA_MODEL,
)


llm = ChatOpenAI(
    base_url=DATA_BASE_URL,
    api_key=DASHSCOPE_API_KEY,
    model=DATA_MODEL,
    temperature=0,
)