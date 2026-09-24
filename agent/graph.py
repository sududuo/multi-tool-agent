from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from agent.memory import memory

from agent.state import State

from agent.summary import (
    summary_router,
    summarize_node,
)

from agent.nodes import agent_node

from agent.model import tools


# ============================================================
# 1. 创建 ToolNode
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 2. 创建 Graph
# ============================================================

graph = StateGraph(State)


# ============================================================
# 3. 注册节点
# ============================================================

graph.add_node(
    "summarize",
    summarize_node,
)

graph.add_node(
    "agent",
    agent_node,
)

graph.add_node(
    "tools",
    tool_node,
)


# ============================================================
# 4. START → summarize / agent
# ============================================================

graph.add_conditional_edges(
    START,
    summary_router,
    {
        "summarize": "summarize",
        "agent": "agent",
    },
)


# ============================================================
# 5. summarize → agent
# ============================================================

graph.add_edge(
    "summarize",
    "agent",
)


# ============================================================
# 6. agent → tools / END
# ============================================================

graph.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "__end__": END,
        "tools": "tools",
    },
)


# ============================================================
# 7. tools → agent
# ============================================================

graph.add_edge(
    "tools",
    "agent",
)



# ============================================================
# 8. 编译 Graph
# ============================================================

app = graph.compile(
    checkpointer=memory,
)