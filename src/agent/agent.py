#! /usr/bin/env python3
from agent.model import model
from tools.tools import get_current_weather, get_current_time
from graph.state import State
from langchain_core.messages import AIMessage, HumanMessage


# 定义一个调用大模型的函数
def call_model(state: State):
    messages = state["messages"]
    # 添加系统消息，指导模型处理单一任务
    system_message = {
        "role": "system",
        "content": """你是一个用于回答用户问题的助手。遵循以下指南：

1. 每次只处理一个任务，不要组合处理多个任务。
2. 如果问题是寻求信息或知识，优先直接回答问题。
3. 如果任务需要获取实时信息（如天气或时间），请使用相应工具。
4. 使用工具后，应该基于工具结果提供一个完整、有用的回答。
5. 保持回答简洁、清晰和有用。

当你使用工具时，先解释为什么你需要使用该工具，然后使用工具，最后基于工具结果提供完整回答。""",
    }

    # 调整消息，在消息开头添加系统消息提示
    processed_messages = [system_message] + messages

    # 绑定工具
    llm_with_tools = model.bind_tools([get_current_weather, get_current_time])

    # 让模型处理任务
    response = llm_with_tools.invoke(processed_messages)

    # 返回包含响应的新状态
    return {"messages": [response]}
