#! /usr/bin/env python3

from pydantic import BaseModel
from typing import List, Dict, Any, Literal

from langgraph.graph import StateGraph, MessagesState, START, END
from graph.state import State
from tools.tools import get_current_weather, get_current_time
from langgraph.prebuilt import ToolNode
from agent.agent import call_model
from typing import Literal
from langchain_core.messages import AIMessage, ToolMessage
from graph.state import State

tools = [get_current_weather, get_current_time]
tool_node = ToolNode(tools)


# 定义一个函数确定是否继续执行
def should_continue(state: State) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    
    # 检查最后一条消息是否是AI消息且包含工具调用
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # 如果没有工具调用或者已经处理完工具调用，结束执行
    return END

# 处理工具调用结果
def process_tool_results(state: State) -> State:
    messages = state['messages']
    # 检查最后一条消息是否是工具消息
    if messages and isinstance(messages[-1], ToolMessage):
        # 获取工具调用的结果
        tool_result = messages[-1].content
        # 将工具调用结果添加到消息列表中
        messages.append(AIMessage(content=tool_result))
    
    # 返回当前状态，让模型处理工具调用结果
    return state

# 创建状态图实例，使用State类型作为状态模式
parallelWorkflow = StateGraph(MessagesState)
# 添加节点
parallelWorkflow.add_node("agent", call_model)
parallelWorkflow.add_node("tools", tool_node)
parallelWorkflow.add_node("process_results", process_tool_results)

# 设置图的边和流程
parallelWorkflow.add_edge(START, "agent")
parallelWorkflow.add_conditional_edges("agent", should_continue)
# 工具调用后，处理结果，再传给模型
parallelWorkflow.add_edge("tools", "process_results")
parallelWorkflow.add_edge("process_results", END)

# 编译工作流以便执行
parallelWorkflow = parallelWorkflow.compile()
