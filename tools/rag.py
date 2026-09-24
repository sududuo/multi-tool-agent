from config import (
    DB_PATH,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL
)
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.documents import Document
import logging
import dashscope
from services.embeddings import embeddings
from langchain_core.retrievers import BaseRetriever

dashscope.api_key = DASHSCOPE_API_KEY
dashscope.base_http_api_url = DASHSCOPE_BASE_URL

logger = logging.getLogger(__name__)

_retriever: BaseRetriever | None = None
def get_retriever() -> BaseRetriever | None:

    global _retriever

    if _retriever is not None:
        return _retriever

    if not DB_PATH.exists():
        return None

    db = Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embeddings,
    )

    _retriever = db.as_retriever(
        search_kwargs={"k": 10}
    )

    return _retriever


RERANK_TOP_N = 3
def rerank_documents(
        query: str,
        documents: list[Document],
) -> list[Document]:
    texts = [
        document.page_content
        for document in documents
    ]

    try:
        response = dashscope.TextReRank.call(
            model="qwen3.7-text-rerank",
            query=query,
            documents=texts,
            top_n=RERANK_TOP_N,
            instruct=(
                "Given a web search query, "
                "retrieve relevant passages that answer the query."
            ),
        )



        if response.status_code != 200:
            logger.warning(
                "Reranker 请求失败：code=%s message=%s",
                response.code,
                response.message,
            )

            return documents[:RERANK_TOP_N]
        result = response.output["results"]


        if not result:
            # 记录 warning
            logger.warning(
                "Reranker 未返回排序结果，降级使用 Retriever 结果"
            )

            # fallback
            return documents[:RERANK_TOP_N]

        logger.info(
            "Rerank成功：候选数=%s，返回数=%s",
            len(documents),
            len(result),
        )
        reranked_documents = []

        for res in result:
            index = res["index"]
            score= res["relevance_score"]
            reranked_documents.append(
                documents[index]
            )


            logger.debug(
                "Rerank结果：index=%s score=%.4f content=%s",
                index,
                score,
                documents[index].page_content[:50],
            )


        return reranked_documents


    except Exception as e:
        logger.warning(
            "Reranker API访问失败，降级使用 Retriever 结果：%s",
            e,
        )

        return documents[:RERANK_TOP_N]

@tool
def search_company_knowledge(query: str) -> str:
    """
    当用户查询有关公司的信息时，使用此工具。

    Args:
        query: 用户关于公司的问题。

    Returns:
        从公司知识库检索到的相关内容。
    """

    try:
        retriever = get_retriever()

        if retriever is None:
            return "公司知识库尚未初始化，请先构建知识库。"

        documents = retriever.invoke(query)

        if not documents:
            return "知识库中没有检索到相关信息。"

        reranked_documents=rerank_documents(query, documents)

        result = "\n".join(
            document.page_content
            for document in reranked_documents
        )

        return result


    except Exception:

        logger.exception(

            "RAG 检索流程发生异常"

        )

        return "抱歉，公司知识库查询出现异常，请稍后再试。"
