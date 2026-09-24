from typing import Annotated, NotRequired, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class State(TypedDict):

    # LangGraph 自动合并消息
    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]

    # 历史摘要
    summary: NotRequired[str]

    # 上次摘要时的消息 Token 总数
    summary_token_count: NotRequired[int]

    # 上次摘要时处理到的消息位置
    summary_message_count: NotRequired[int]

