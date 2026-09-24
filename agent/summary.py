from agent.llm import llm
from config import (
    SUMMARY_TOKEN_THRESHOLD,
    SUMMARY_INTERVAL_TOKEN,
)

from agent.state import State
from agent.context import count_message_tokens

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

def summarize_node(state: State):
    """
    使用 LLM 总结上次摘要之后新增的消息。

    第一次摘要时处理全部已有消息。
    后续摘要时处理新增消息，并参考旧摘要。
    """

    messages = state["messages"]

    old_summary = state.get(
        "summary",
        "",
    )

    summary_message_count = state.get(
        "summary_message_count",
        0,
    )

    # 只取上次摘要之后新增的消息
    new_messages = messages[
        summary_message_count:
    ]

    summary_prompt = [

        SystemMessage(
            content="""
你是一个对话摘要助手。

请根据旧摘要和新增对话，
生成一份更新后的完整摘要。

需要保留：

1. 用户的长期目标和偏好
2. 已经完成的任务
3. 正在进行的任务
4. 重要的技术结论
5. 后续需要保留的上下文

要求：

- 不要编造不存在的信息。
- 保留重要技术细节。
- 删除无关的闲聊。
- 输出简洁、准确的摘要。
- 新摘要必须包含旧摘要中仍然重要的信息。
"""
        ),

        HumanMessage(
            content=f"""
旧摘要：

{old_summary}

新增对话：

{new_messages}
"""
        ),
    ]

    response = llm.invoke(
        summary_prompt
    )

    current_token_count = (
        count_message_tokens(messages)
    )

    return {

        "summary": response.content,

        # 记录当前已经处理到的消息位置
        "summary_message_count": len(messages),

        # 记录本次摘要时的总 Token 数
        "summary_token_count": current_token_count,
    }

def summary_router(state: State):
    """
    判断本次运行是否需要生成摘要。
    """

    messages = state["messages"]

    current_token_count = (
        count_message_tokens(messages)
    )

    summary_token_count = state.get(
        "summary_token_count",
        0,
    )

    new_token_count = (
        current_token_count
        - summary_token_count
    )

    if (
        current_token_count
        >= SUMMARY_TOKEN_THRESHOLD
        and new_token_count
        >= SUMMARY_INTERVAL_TOKEN
    ):
        return "summarize"

    return "agent"