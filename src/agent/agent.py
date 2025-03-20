#! /usr/bin/env python3
from agent.model import model
from tools.tools import get_current_weather, get_current_time
from graph.state import State
# 定义一个调用大模型的函数
def call_model(state: State):
    messages = state['messages']
    # 绑定工具
    llm_with_tools = model.bind_tools([get_current_weather, get_current_time])
    response = llm_with_tools.invoke(messages)

    print("response",response)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}
