from langchain_openai import OpenAIEmbeddings

from config import (
    DASHSCOPE_API_KEY,
    DATA_BASE_URL,
)


embeddings = OpenAIEmbeddings(
    api_key=DASHSCOPE_API_KEY,
    base_url=DATA_BASE_URL,
    model="text-embedding-v3",
    check_embedding_ctx_length=False,
    chunk_size=10,
)