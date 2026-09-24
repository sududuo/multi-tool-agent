from agent.llm import llm

from tools.weather import get_weather
from tools.rag import search_company_knowledge


tools = [
    get_weather,
    search_company_knowledge,
]

llm_with_tools = llm.bind_tools(tools)