#! /usr/bin/env python3

from typing import List, Dict, Any, Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
import json
import asyncio

from graph.state import State, AnalyzeMessageTask
from tools.tools import get_current_weather, get_current_time
from agent.agent import call_model
from agent.model import model
from graph.generate_img import generate_img
# 初始化工具节点
tools = [get_current_weather, get_current_time]
tool_node = ToolNode(tools)

# 任务分析提示模板
TASK_ANALYSIS_PROMPT = PromptTemplate.from_template(
    """你是一个专业的任务分析专家。请仔细分析用户消息中包含的所有任务。

任务分析要求:
1. 将用户消息拆分为独立的子任务
2. 每个任务需要包含:
   - 任务ID: task_1, task_2 等
   - 具体任务内容
   - 任务类型: query(查询) 或 action(操作)
   - 优先级: 1-5 (5最高)
   - 是否需要工具: true/false
   - 工具名称: get_current_weather 或 get_current_time

输出格式要求:
必须是标准JSON数组,每个任务对象包含以下字段:
[
  {{
    "id": "task_1",
    "content": "查询上海天气", 
    "type": "query",
    "priority": 5,
    "requires_tool": true,
    "tool_call": "get_current_weather"
  }}
]

注意:
- 优先级根据任务紧急程度和依赖关系确定
- 工具调用必须使用系统支持的工具名称
- JSON格式必须严格符合规范,不要添加其他内容

用户消息: {user_content}"""
)

async def analyze_message(state: State) -> State:
    """分析用户消息中的多个任务，并结构化提取出这些任务"""
    messages = state["messages"]
    last_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    
    if not last_message:
        return state

    try:
        # 调用模型分析任务
        agent = TASK_ANALYSIS_PROMPT | model
        result = await agent.ainvoke({"user_content": last_message.content})
        
        # 清理和解析JSON
        json_str = result.content.strip()
        start, end = json_str.find('['), json_str.rfind(']') + 1
        if start != -1 and end > 0:
            json_str = json_str[start:end]
            
        data = json.loads(json_str)
        tasks = [AnalyzeMessageTask.from_json(task_data) for task_data in data]
        tasks.sort(key=lambda x: x.priority, reverse=True)
        
        state.update({"tasks": tasks})
        return state

    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}\n问题字符串: {result.content}")
        return state
    except Exception as e:
        print(f"任务分析错误: {e}\n完整错误信息: {str(e)}")
        return state

async def process_task(state: State) -> State:
    """处理当前任务并更新状态"""
    tasks = state.get("tasks", [])
    current_index = state.get("current_task_index", 0)

    if not tasks or current_index >= len(tasks):
        return state

    current_task = tasks[current_index]
    print(f"处理任务 {current_index + 1}/{len(tasks)}: {current_task}")

    if not current_task.requires_tool:
        return {"messages": [HumanMessage(content=current_task.content)]}

    # 处理工具调用
    tool_func = {
        "get_current_weather": get_current_weather,
        "get_current_time": get_current_time
    }.get(current_task.tool_call)

    if tool_func:
        try:
            if current_task.tool_call == "get_current_weather":
                location = current_task.content.split()[-1]
                result = await tool_func.ainvoke(input=location)
            else:
                result = await tool_func.ainvoke(input={})
            current_task.result = result
        except Exception as e:
            current_task.result = f"工具调用失败: {str(e)}"

    tool_instruction = f"请使用{current_task.tool_call}工具来完成以下任务: {current_task.content}"
    return {"messages": [HumanMessage(content=tool_instruction)]}

async def save_task_result(state: State) -> State:
    """保存当前任务的处理结果并更新状态"""
    messages = state["messages"]
    tasks = state.get("tasks", [])
    current_index = state.get("current_task_index", 0)
    task_results = state.get("task_results", [])

    if not tasks or current_index >= len(tasks):
        return state

    # 收集结果
    result_parts = []
    ai_responses = [msg.content for msg in messages 
                   if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None)]
    tool_responses = [f"工具【{getattr(msg, 'name', '未知工具')}】执行结果: {msg.content}"
                     for msg in messages if isinstance(msg, ToolMessage)]
    
    result_parts.extend(ai_responses)
    result_parts.extend(tool_responses)
    result = "\n\n".join(result_parts).strip() or "没有找到相关回复"

    # 更新任务结果
    current_task = tasks[current_index]
    if not current_task.result:
        current_task.result = result

    return {
        **state,
        "tasks": tasks,
        "task_results": [*task_results, {"task": current_task, "result": current_task.result}],
        "current_task_index": current_index + 1,
        "tool_invocations": 0
    }

async def has_more_tasks(state: State) -> Literal["process_next_task", "assemble_response"]:
    """判断是否有更多任务需要处理"""
    tasks = state.get("tasks", [])
    current_index = state.get("current_task_index", 0)
    return "process_next_task" if current_index < len(tasks) else "assemble_response"

async def should_continue(state: State) -> Literal["tools", END]: # type: ignore
    """确定是否继续执行工具调用"""
    messages = state["messages"]
    last_message = messages[-1]
    tool_invocations = state.get("tool_invocations", 0)

    if tool_invocations >= 5:
        return END

    if (isinstance(last_message, AIMessage) and 
        hasattr(last_message, "tool_calls") and 
        last_message.tool_calls):
        state["tool_invocations"] = tool_invocations + 1
        return "tools"

    return END

async def process_tool_results(state: State) -> State:
    """处理工具调用结果"""
    messages = state["messages"]
    tool_message = next((msg for msg in reversed(messages) 
                        if isinstance(msg, ToolMessage)), None)
    return state

async def assemble_response(state: State) -> State:
    """组装所有任务处理结果为最终响应"""
    task_results = state.get("task_results", [])
    original_messages = state.get("original_messages", [])

    if not task_results:
        return state

    # 提取有效的任务结果
    all_results = [
        result['task'].result 
        for result in task_results
        if result.get('task') and result['task'].result
    ]

    if not all_results:
        return state

    try:
        # 构建提示消息并生成总结
        prompt = PromptTemplate.from_template(
            """你是一个专业的助手。请根据以下多个查询结果生成一个完整的回复。

要求:
1. 保持专业、准确的语气
2. 确保信息的连贯性和完整性
3. 如果结果中包含多个工具调用,需要合理组织它们之间的关系
4. 使用清晰的结构,适当分段
5. 只包含查询结果中的信息,不要添加其他内容
6. 使用中文回答

查询结果:
{content}

请基于以上结果生成一个专业的回复。"""
        )
        
        final_message = await (prompt | model).ainvoke({"content": "\n\n".join(all_results)})

        print(final_message)
        
        # 确保返回AIMessage类型
        final_message = (
            final_message if isinstance(final_message, AIMessage)
            else AIMessage(content=str(final_message))
        )
        
    except Exception as e:
        print(f"生成总结时出现错误: {e}")
        final_message = AIMessage(content="抱歉,生成总结时出现错误")

    # 更新并返回状态
    return {
        **state,
        "messages": [*original_messages, final_message] if original_messages else [final_message]
    }

async def save_original_messages(state: State) -> State:
    """保存原始消息历史"""
    new_state = state.copy()
    new_state["original_messages"] = state["messages"]
    return new_state



async def workflow_builder(state: State) -> StateGraph:
    """构建工作流"""
    workflow = StateGraph(state)

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
    workflow.add_conditional_edges(
        "save_task_result",
        has_more_tasks,
        {"process_next_task": "process_task", "assemble_response": "assemble_response"},
    )
    workflow.add_edge("assemble_response", END)

    # 编译工作流
    return workflow.compile()