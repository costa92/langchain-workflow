#! /usr/bin/env python3

from pydantic import BaseModel
from typing import List, Dict, Any, Literal

from langgraph.graph import StateGraph, MessagesState, START, END
from graph.state import State
from tools.tools import get_current_weather, get_current_time
from langgraph.prebuilt import ToolNode
from agent.agent import call_model
from typing import Literal
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from graph.state import State

tools = [get_current_weather, get_current_time]
tool_node = ToolNode(tools)


# 定义一个分析消息的函数，把信息提出出来，分析多个任务
def analyze_message(state: State) -> State:
    """
    分析用户消息中的多个任务，并结构化提取出这些任务
    
    参数:
        state: 当前状态，包含消息历史
        
    返回:
        更新后的状态，包含结构化的任务信息
    """
    messages = state['messages']
    # 获取最后一条消息，应该是用户消息
    last_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    
    if not last_message:
        return state
    
    # 提取用户消息内容
    user_content = last_message.content
    
    # 简单的任务分割逻辑：按问号、逗号或句号分割
    task_delimiters = ['?', '？', ',', '，', '.', '。']
    tasks = []
    
    # 临时存储当前任务文本
    current_task = ""
    
    # 遍历消息内容，按分隔符分割任务
    for char in user_content:
        current_task += char
        if char in task_delimiters and current_task.strip():
            tasks.append(current_task.strip())
            current_task = ""
    
    # 添加最后一个任务（如果有）
    if current_task.strip():
        tasks.append(current_task.strip())
    
    # 任务去重
    unique_tasks = []
    for task in tasks:
        if task not in unique_tasks:
            unique_tasks.append(task)
    
    # 将解析后的任务添加到状态中
    state['tasks'] = unique_tasks
    state['current_task_index'] = 0
    state['task_results'] = []
    
    return state

# 处理单个任务
def process_task(state: State) -> State:
    """处理当前任务并更新状态"""
    tasks = state.get('tasks', [])
    current_index = state.get('current_task_index', 0)
    
    if not tasks or current_index >= len(tasks):
        return state
    
    # 获取当前任务
    current_task = tasks[current_index]
    
    # 输出当前正在处理的任务（调试用）
    print(f"处理任务 {current_index + 1}/{len(tasks)}: {current_task}")
    
    # 创建一个新的消息列表，只包含当前任务
    # 这样LLM只需要处理一个具体的任务
    return {"messages": [HumanMessage(content=current_task)]}

# 保存任务结果
def save_task_result(state: State) -> State:
    """保存当前任务的处理结果"""
    messages = state['messages']
    tasks = state.get('tasks', [])
    current_index = state.get('current_task_index', 0)
    task_results = state.get('task_results', [])
    
    # 确保我们有任务要处理
    if not tasks or current_index >= len(tasks):
        return state
    
    # 提取处理结果
    result = ""
    
    # 获取非工具调用的AI回复
    final_ai_messages = []
    for msg in messages:
        if isinstance(msg, AIMessage) and not (hasattr(msg, "tool_calls") and msg.tool_calls):
            final_ai_messages.append(msg.content)
    
    # 获取工具调用结果
    tool_results = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_name = msg.name if hasattr(msg, 'name') else '未知工具'
            tool_results.append(f"工具【{tool_name}】执行结果: {msg.content}")
    
    # 如果有AI消息回复，添加到结果中
    if final_ai_messages:
        result += "\n".join(final_ai_messages)
    
    # 如果有工具调用结果，添加到结果中
    if tool_results:
        if result:
            result += "\n\n"
        result += "\n".join(tool_results)
        
    # 如果结果为空，提供默认消息
    if not result.strip():
        result = "没有找到相关回复"
    
    # 保存当前任务和其结果
    task_results.append({
        "task": tasks[current_index],
        "result": result
    })
    
    # 重置工具调用计数
    state['tool_invocations'] = 0
    
    # 输出完成处理的任务（调试用）
    print(f"完成任务 {current_index + 1}: {tasks[current_index]}")
    
    # 更新状态
    new_state = state.copy()
    new_state['task_results'] = task_results
    new_state['current_task_index'] = current_index + 1
    
    return new_state

# 检查是否有更多任务
def has_more_tasks(state: State) -> Literal["process_next_task", "assemble_response"]:
    """判断是否有更多任务需要处理"""
    tasks = state.get('tasks', [])
    current_index = state.get('current_task_index', 0)
    
    if current_index < len(tasks):
        return "process_next_task"
    else:
        return "assemble_response"

# 定义一个函数确定是否继续执行
def should_continue(state: State) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    tool_invocations = state.get('tool_invocations', 0)
    
    # 最多允许5次工具调用，防止无限循环
    if tool_invocations >= 5:
        return END
    
    # 检查最后一条消息是否是AI消息且包含工具调用
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # 增加工具调用计数
        state['tool_invocations'] = tool_invocations + 1
        return "tools"
    
    # 如果没有工具调用或者已经处理完工具调用，结束执行
    return END

# 处理工具调用结果
def process_tool_results(state: State) -> State:
    """处理工具调用结果"""
    messages = state['messages']
    # 获取最后一条工具消息（如果有）
    tool_message = next((msg for msg in reversed(messages) if isinstance(msg, ToolMessage)), None)
    
    if tool_message:
        # 返回原始消息列表，但确保工具消息在消息列表中
        return state
    
    # 如果没有工具消息，直接返回原始状态
    return state

# 组装最终响应
def assemble_response(state: State) -> State:
    """组装所有任务处理结果为最终响应"""
    task_results = state.get('task_results', [])
    original_messages = state.get('original_messages', [])
    
    if not task_results:
        return state
    
    # 创建综合响应
    combined_response = "我已经处理了您的多个请求，以下是回复：\n\n"
    
    for i, result in enumerate(task_results, 1):
        combined_response += f"问题 {i}: {result['task']}\n"
        combined_response += f"回答: {result['result']}\n\n"
    
    # 创建一个包含最终综合响应的AIMessage
    final_message = AIMessage(content=combined_response)
    
    # 如果有原始消息历史，则保留它并添加最终回复
    if original_messages:
        return {"messages": original_messages + [final_message]}
    else:
        # 否则只返回最终回复
        return {"messages": [final_message]}

# 保存原始消息
def save_original_messages(state: State) -> State:
    """保存原始消息历史"""
    new_state = state.copy()
    new_state['original_messages'] = state['messages']
    return new_state

# 创建状态图实例，使用State类型作为状态模式
workflow = StateGraph(State)

# 添加节点
workflow.add_node("analyze_message", analyze_message)
workflow.add_node("process_task", process_task)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("process_tool_results", process_tool_results)
workflow.add_node("save_task_result", save_task_result)
workflow.add_node("assemble_response", assemble_response)
workflow.add_node("save_original_messages", save_original_messages)

# 设置图的边和流程
workflow.add_edge(START, "save_original_messages")
workflow.add_edge("save_original_messages", "analyze_message")
workflow.add_edge("analyze_message", "process_task")
workflow.add_edge("process_task", "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "process_tool_results")
workflow.add_edge("process_tool_results", "agent")
workflow.add_edge("agent", "save_task_result")
workflow.add_conditional_edges("save_task_result", has_more_tasks, {
    "process_next_task": "process_task", 
    "assemble_response": "assemble_response"
})
workflow.add_edge("assemble_response", END)

# 编译工作流以便执行
parallelWorkflow = workflow.compile()

# INSERT_YOUR_REWRITE_HERE
from IPython.display import display
from PIL import Image as PILImage
import io
image_data = parallelWorkflow.get_graph().draw_mermaid_png()
image = PILImage.open(io.BytesIO(image_data))
image.save("output.png")