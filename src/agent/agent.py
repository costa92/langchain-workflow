#! /usr/bin/env python3
from agent.model import model
from tools.tools import get_current_weather, get_current_time
from graph.state import State
from langchain_core.messages import AIMessage, HumanMessage
import logging

async def call_model(state: State) -> State:
    """调用大语言模型处理任务并返回响应"""
    messages = state.get("messages", [])
    
    if len(messages) < 2:
        logging.warning("消息列表不足，无法去除第一条消息")
        return state

    last_message = messages[1:]  # 去除第一条消息
    logging.info(f"Processing message: {last_message}")

    system_message = {
        "role": "system",
        "content": """你是一个智能助手,专注于提供准确和有帮助的回答。

角色定位:
- 专业的问题解答助手
- 保持客观专业的语气
- 确保回答准确可靠

回答原则:
1. 工具调用
- 需要实时数据时必须使用对应工具
- 天气查询使用 get_current_weather
- 时间查询使用 get_current_time
- 不生成虚假数据
- 工具调用失败需告知用户

2. 结果处理
- 直接使用工具返回结果
- 保持回答流畅自然
- 缺少信息时明确告知

3. 回答质量
- 准确性: 基于工具返回结果
- 相关性: 围绕用户查询
- 完整性: 包含必要信息
- 简洁性: 避免冗余

可用工具:
- get_current_weather(location: str): 获取指定地点实时天气
- get_current_time(): 获取当前系统时间

注意事项:
- 不推测未经验证的信息
- 不确定时说明不知道
- 使用工具获取实时数据
- 直接使用工具返回结果"""
    }

    try:
        processed_messages = [system_message] + last_message
        llm_with_tools = model.bind_tools([get_current_weather, get_current_time])
        response = llm_with_tools.invoke(processed_messages)

        logging.info(f"模型返回响应: {response.content}")
        return {"messages": messages[1:] + [response]}
        
    except Exception as e:
        logging.error(f"模型调用出错: {e}", exc_info=True)
        return {"messages": messages[1:] + [AIMessage(content="抱歉,模型调用出现错误,请稍后重试。")]}