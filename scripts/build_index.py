from config import (
    COMPANY_FILE,
    DB_PATH,
)

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.embeddings import embeddings
import shutil

def build_index():
    if DB_PATH.exists():
        print(
            "检测到旧知识库存在，正在删除旧知识库"
        )
        shutil.rmtree(DB_PATH)
        print(
            "删除完成"
        )

    with open(
        COMPANY_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=20,
        chunk_overlap=5,
    )

    documents = splitter.create_documents(
        [text]
    )

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(DB_PATH),
    )
    print(
        f"知识库构建完成，共写入 {len(documents)} 个文档块。"
    )


if __name__ == "__main__":
    build_index()