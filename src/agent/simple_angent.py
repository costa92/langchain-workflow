#! /usr/bin/env python3

import asyncio
import json # Added for potential validation later
from typing import Annotated, List, TypedDict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage

from langgraph.graph import StateGraph, END
from langgraph.graph import add_messages
# Assuming 'tools.tools.calculator' exists and is a LangChain-compatible tool
from tools.tools import calculator # Keep your tool import
from langgraph.prebuilt import ToolNode, tools_condition

# Define the structure for the final output JSON (remains the same)
class AnalysisReport(TypedDict):
    analysis_message: str
    user_input: str

# Refined Agent State using TypedDict
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str # Store the original user input
    # Removed 'analysis_result' as the final result is in the last message

# System prompt refined to explicitly mention the tool and formatting
# Note: Changed role slightly to emphasize using tools IF needed.
# Adjusted formatting instructions slightly for clarity.
systemPrompt = """## 角色
你是一个智能计算助手。你的主要任务是根据用户输入进行计算。
你配备了一个 'calculator' 工具来执行数学运算。

## 任务
1.  理解用户的计算请求。
2.  对于任何计算请求，你**必须**使用 'calculator' 工具来执行计算。即使是简单的计算也要使用工具。
3.  根据工具返回的计算结果，生成最终响应。

## 输出格式
你的**最终**输出**必须**是一个 JSON 对象，严格遵循以下格式，不包含任何其他解释性文字或标记（如 ```json ... ```）：
{
  "analysis_message": "这里是计算结果",
  "user_input": "这里是用户最初输入的内容"
}

## 指导
- 对于所有计算请求，无论简单或复杂，都必须使用 'calculator' 工具。
- 调用 'calculator' 工具时，提供清晰的数学表达式。
- 工具返回结果后，将该结果格式化到最终的 JSON 响应中。
- 确保 `analysis_message` 字段只包含计算的最终答案或结果。
- 确保 `user_input` 字段包含用户未经修改的原始输入。

"""

class SimpleAgent:
    """
    优化的代理类，使用 LangGraph 的标准工具使用模式与LLM交互生成分析报告。
    """

    def __init__(self, llm: BaseChatModel):
        """
        初始化 OptimizedAgent

        Args:
            llm (BaseChatModel): 语言模型实例
        """
        self.llm = llm
        # Ensure tools are provided in a list
        self.tools = [calculator]
        # Bind tools to the LLM correctly
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        # Build the graph during initialization
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建状态图 (采用标准的 Agent->Action->Agent 循环)"""
        graph = StateGraph(AgentState)

        # Node that calls the LLM (Agent)
        graph.add_node("agent", self.call_model)
        # Node that executes tools (Action)
        graph.add_node("action", ToolNode(self.tools))

        # Entry point is the agent node
        graph.set_entry_point("agent")

        # Conditional edge: Determine whether to call tools or end
        # Uses the built-in 'tools_condition'
        graph.add_conditional_edges(
            "agent",
            tools_condition,
            # If tools are called, go to 'action'. Otherwise, end.
            {
                "tools": "action",
                END: END
            }
        )

        # Edge from action (tool execution) back to the agent to process the results
        graph.add_edge("action", "agent")

        # Compile the graph
        return graph.compile()

    async def call_model(self, state: AgentState) -> dict:
        """
        调用LLM并判断是否需要使用工具。
        
        Args:
            state (AgentState): 当前状态，包含消息历史

        Returns:
            dict: 包含新消息的状态更新，由LangGraph通过add_messages自动合并
        """
        print("---Calling LLM with Tools---")
        messages = state['messages']
        
        # 使用绑定了工具的LLM进行调用
        response = await self.llm_with_tools.ainvoke(messages)
        
        # 打印调试信息
        print(f"---LLM Response: {response} ---")
        print(f"---LLM Response Type: {type(response).__name__} ---")
        
        # 检查响应是否包含工具调用
        if isinstance(response, AIMessage) and hasattr(response, 'tool_calls') and response.tool_calls:
            print("---Tool Call Detected---")
            for tool_call in response.tool_calls:
                print(f"Tool: {tool_call['name']}, Arguments: {tool_call['args']}")
                # print(tool_call)
        else:
            print("---Direct Response (No Tool Call)---")
            
        # 返回响应消息以更新状态
        return {"messages": [response]}
    async def run(self, content: str) -> Optional[str]:
        """
        运行代理进行推理, 生成分析报告的JSON字符串。

        Args:
            content (str): 用户输入内容

        Returns:
            Optional[str]: LLM生成的最终分析报告JSON字符串, or None if failed.
        """
        print(f"---Running Agent with Input: {content}---")
        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=systemPrompt),
                HumanMessage(content=content)
            ],
            "user_input": content,
        }
        # 使用astream方法获取执行过程中的事件流
        result_stream = self.graph.astream(initial_state)
        return result_stream
