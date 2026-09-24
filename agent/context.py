from agent.llm import llm
from langchain_core.messages import BaseMessage,HumanMessage


def count_message_tokens(
    messages: list[BaseMessage],
) -> int:
    """
    计算消息列表的 Token 数量。

    优先使用 LLM 提供的消息 Token 统计方法。
    """

    if not messages:
        return 0

    try:

        return llm.get_num_tokens_from_messages(
            messages
        )

    except Exception:

        # 备用的粗略估算。
        # 这不是严格 Token 统计。
        total_chars = sum(
            len(
                str(
                    getattr(
                        message,
                        "content",
                        "",
                    )
                )
            )
            for message in messages
        )

        return total_chars // 2

def get_safe_recent_messages(
    messages: list[BaseMessage],
    token_limit: int,
) -> list[BaseMessage]:
    """
    按完整对话轮次保留近期消息。

    一轮对话从 HumanMessage 开始，
    到下一条 HumanMessage 之前结束。

    从最新轮次开始向前选择，
    尽量不超过 Token 预算。

    注意：
    这是学习版的轮次裁剪逻辑，
    不是对所有复杂工具调用情况的完整验证。
    """

    if not messages:
        return []

    # --------------------------------------------------------
    # 第一步：找到所有 HumanMessage 的索引
    # --------------------------------------------------------

    human_indexes = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    ]

    if not human_indexes:
        return []

    # --------------------------------------------------------
    # 第二步：按照 HumanMessage 划分对话轮次
    # --------------------------------------------------------

    conversation_turns = []

    for index, start_index in enumerate(
        human_indexes
    ):

        # 如果后面还有 HumanMessage，
        # 下一条 HumanMessage 就是当前轮次的结束位置
        if index + 1 < len(human_indexes):

            end_index = human_indexes[index + 1]

        # 最后一轮一直取到消息列表结尾
        else:

            end_index = len(messages)

        turn = messages[
            start_index:end_index
        ]

        conversation_turns.append(turn)

    # --------------------------------------------------------
    # 第三步：从最新轮次开始向前选择
    # --------------------------------------------------------

    selected_turns = []

    current_tokens = 0

    for turn in reversed(conversation_turns):

        turn_tokens = count_message_tokens(turn)

        # 如果最新一轮单独就超过预算，
        # 仍然保留最新一轮，避免当前问题完全丢失
        if (
            not selected_turns
            and turn_tokens > token_limit
        ):

            selected_turns.append(turn)

            break

        # 如果加入当前轮次后超过预算，
        # 就停止添加更早的轮次
        if (
            current_tokens + turn_tokens
            > token_limit
        ):

            break

        selected_turns.append(turn)

        current_tokens += turn_tokens

    # --------------------------------------------------------
    # 第四步：恢复时间顺序
    # --------------------------------------------------------

    selected_turns.reverse()

    # --------------------------------------------------------
    # 第五步：将二维列表合并为一维消息列表
    # --------------------------------------------------------

    recent_messages = []

    for turn in selected_turns:

        recent_messages.extend(turn)

    return recent_messages