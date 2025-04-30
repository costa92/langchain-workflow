#! /usr/bin/env python3
import logging
from typing import Annotated, List, Dict, Any, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from llm.llm import init_deepseek
from tools.tools import get_current_weather, get_current_time
from langgraph.prebuilt import ToolNode
import json
from typing_extensions import TypedDict
from dataclasses import dataclass

# 导入提示词模板
from prompts.task_analysis import TASK_ANALYSIS_PROMPT
from prompts.tool_analysis import TOOL_ANALYSIS_PROMPT
from prompts.response_generation import TOOL_SUMMARY_PROMPT, MODEL_SUMMARY_PROMPT


@dataclass
class AnalyzeMessageTask:
    """任务分析结果的数据类"""
    id: str
    content: str
    requires_tool: bool
    tool_call: str
    result: str = ""
    processed: bool = False  # 标记任务是否已处理
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "AnalyzeMessageTask":
        return cls(**json_data)

def merge_tasks(old: List[AnalyzeMessageTask], new: List[AnalyzeMessageTask]) -> List[AnalyzeMessageTask]:
    if old is None:
        return new or []
    if new is None:
        return old
    old_ids = {t.id for t in old}
    return old + [t for t in new if t.id not in old_ids]

def merge_task_index(old: int, new: int) -> int:
    if old is None:
        return new or 0
    if new is None:
        return old
    return new

def merge_task_results(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    if old is None:
        return new or {}
    if new is None:
        return old
    return {**old, **new}

def merge_processed_tasks(old: set, new: set) -> set:
    if old is None:
        return new or set()
    if new is None:
        return old
    return old | new

def merge_tool_invocations(old: int, new: int) -> int:
    if old is None:
        return new or 0
    if new is None:
        return old
    return new

def merge_original_messages(old: List[AnyMessage], new: List[AnyMessage]) -> List[AnyMessage]:
    if old is None:
        return new or []
    if new is None:
        return old
    return new

class TaskAnalyzerState(TypedDict):
    """任务分析器状态类型定义"""
    messages: Annotated[List[AnyMessage], add_messages]
    chat_analyzer_messages: Annotated[List[AnyMessage], add_messages]
    tasks: Annotated[List[AnalyzeMessageTask], merge_tasks]  # 待处理任务列表
    current_task_index: Annotated[int, merge_task_index]  # 当前处理任务索引
    task_results: Annotated[Dict[str, Any], merge_task_results]  # 任务处理结果字典
    processed_tasks: Annotated[set, merge_processed_tasks]  # 已处理任务集合
    tool_invocations: Annotated[int, merge_tool_invocations]  # 工具调用次数
    original_messages: Annotated[List[AnyMessage], merge_original_messages]  # 原始消息历史


class TaskAnalyzerAgent:
    """任务分析代理类,负责分析用户输入的任务并提供相应的响应"""

    def __init__(self, llm: BaseChatModel = None):
        """初始化任务分析代理"""
        self.llm = llm or init_deepseek()
        self.tools = [get_current_weather, get_current_time]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)
        self.graph = self.build_graph()
        logging.info("任务分析代理初始化完成")

    async def agent_decision(self, state: TaskAnalyzerState) -> Dict[str, Any]:
        """代理决策函数,处理当前状态并返回下一步操作"""
        try:
            messages = state.get("messages", [])
            if not messages:
                logging.warning("没有消息可处理")
                return {}
            
            last_message = messages[-1]
            logging.info(f"处理消息: {last_message.content if last_message else 'None'}")
          
            if not isinstance(last_message, HumanMessage):
                return {
                    "tool_name": None,
                    "tool_input": last_message.content,
                    "result": None
                }
            
            tool_result = await (TOOL_ANALYSIS_PROMPT | self.llm).ainvoke({
                "input": last_message.content
            })
            tool_name = tool_result.content.strip().lower()
            
            if tool_name not in ["get_current_weather", "get_current_time"]:
                model_response = await self.llm.ainvoke([last_message])
                return {
                    "tool_name": None,
                    "tool_input": last_message.content,
                    "result": model_response.content
                }
            
            return {
                "tool_name": tool_name,
                "tool_input": last_message.content,
                "result": None
            }
            
        except Exception as e:
            logging.error(f"代理决策过程出错: {e}", exc_info=True)
            return {}

    async def analyze_message(self, state: TaskAnalyzerState) -> TaskAnalyzerState:
        """分析用户消息中的多个任务,并结构化提取出这些任务"""
        current_index = state.get("current_task_index", 0)
        current_tasks = state.get("tasks", [])
        processed_tasks = state.get("processed_tasks", set())

        if current_tasks and current_index < len(current_tasks):
            logging.info(f"当前还有任务在处理: 进度 {current_index + 1}/{len(current_tasks)}")
            return state
            
        messages = state["messages"]
        last_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        
        if not last_message:
            logging.warning("未找到用户消息")
            return state

        try:
            logging.info(f"开始分析新消息: {last_message.content}")
            result = await (TASK_ANALYSIS_PROMPT | self.llm).ainvoke({
                "user_content": last_message.content
            })
            
            json_str = result.content.strip()
            start, end = json_str.find('['), json_str.rfind(']') + 1
            if start != -1 and end > 0:
                json_str = json_str[start:end]
                
            data = json.loads(json_str)
            new_tasks = [AnalyzeMessageTask.from_json(task_data) for task_data in data]
            new_tasks = [task for task in new_tasks if task.id not in processed_tasks]
            
            if not new_tasks:
                logging.info("没有新的待处理任务")
                return state
                
            logging.info(f"分析出 {len(new_tasks)} 个新任务")
            for task in new_tasks:
                logging.info(f"- {task}")
            
            return {
                **state,
                "tasks": new_tasks,
                "current_task_index": 0,
                "task_results": {},
                "processed_tasks": processed_tasks
            }

        except json.JSONDecodeError as e:
            logging.error(f"JSON解析错误: {e}")
            return state
        except Exception as e:
            logging.error(f"任务分析错误: {e}")
            return state

    async def process_task(self, state: TaskAnalyzerState) -> TaskAnalyzerState:
        """处理当前任务并更新状态"""
        tasks = state.get("tasks", [])
        current_index = state.get("current_task_index", 0)
        processed_tasks = state.get("processed_tasks", set())

        if not tasks or current_index >= len(tasks):
            return state

        current_task = tasks[current_index]
        
        if current_task.id in processed_tasks:
            return {
                **state,
                "current_task_index": current_index + 1
            }
        
        logging.info(f"处理任务 {current_index + 1}/{len(tasks)}: {current_task}")

        tool_func = {
            "get_current_weather": get_current_weather,
            "get_current_time": get_current_time
        }.get(current_task.tool_call)

        if current_task.requires_tool and tool_func:
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
            return {
                **state,
                "messages": [*state.get("messages", []), HumanMessage(content=tool_instruction)]
            }
        else:
            current_task.result = current_task.content
            return {
                **state,
                "current_task_index": current_index + 1,
                "task_results": {
                    **state.get("task_results", {}),
                    current_task.id: {
                        "task": current_task,
                        "result": current_task.result,
                        "is_tool_result": False
                    }
                },
                "processed_tasks": processed_tasks | {current_task.id}
            }

    async def save_task_result(self, state: TaskAnalyzerState) -> TaskAnalyzerState:
        """保存当前任务的处理结果并更新状态"""
        messages = state.get("messages", [])
        tasks = state.get("tasks", [])
        current_index = state.get("current_task_index", 0)
        task_results = state.get("task_results", {})
        processed_tasks = state.get("processed_tasks", set())
        tool_invocations = state.get("tool_invocations", 0)

        if not tasks or current_index >= len(tasks):
            return state

        current_task = tasks[current_index]
        
        ai_responses = [msg.content for msg in messages[-3:] if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None)]
        tool_responses = [f"工具【{getattr(msg, 'name', '未知工具')}】执行结果: {msg.content}" for msg in messages[-3:] if isinstance(msg, ToolMessage)]

        result = "\n".join(tool_responses) if current_task.requires_tool and tool_responses else "\n".join(ai_responses) if ai_responses else "无结果"

        if not current_task.result:
            current_task.result = result
        
        current_task.processed = True
        processed_tasks.add(current_task.id)
        task_results[current_task.id] = {
            "task": current_task,
            "result": current_task.result,
            "is_tool_result": current_task.requires_tool
        }

        return {
            **state,
            "tasks": tasks,
            "task_results": task_results,
            "current_task_index": current_index + 1,
            "tool_invocations": tool_invocations,
            "processed_tasks": processed_tasks,
            "original_messages": state.get("original_messages", [])
        }

    async def has_more_tasks(self, state: TaskAnalyzerState) -> Literal["process_next_task", "assemble_response"]:
        """判断是否有更多任务需要处理"""
        tasks = state.get("tasks", [])
        current_index = state.get("current_task_index", 0)
        return "process_next_task" if current_index < len(tasks) else "assemble_response"

    async def should_continue(self, state: TaskAnalyzerState) -> Literal["tools", "end"]: 
        """确定是否继续执行工具调用"""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        tool_invocations = state.get("tool_invocations", 0)

        if tool_invocations >= 5:
            logging.warning("已达到最大工具调用次数限制")
            return "end"

        if (isinstance(last_message, AIMessage) and 
            hasattr(last_message, "tool_calls") and 
            last_message.tool_calls):
            return "tools"

        return "end"

    async def assemble_response(self, state: TaskAnalyzerState) -> TaskAnalyzerState:
        """组装所有任务处理结果为最终响应"""
        task_results = state.get("task_results", {})
        original_messages = state.get("original_messages", [])
        messages = state.get("messages", [])

        if not task_results:
            return state

        tool_results = []
        model_results = []
        
        for result in task_results.values():
            if result['task'] and result['task'].result:
                if result.get('is_tool_result'):
                    tool_results.append(result['task'].result)
                else:
                    model_results.append(result['task'].result)

        try:
            final_responses = []
            
            if tool_results:
                tool_summary = await (TOOL_SUMMARY_PROMPT | self.llm).ainvoke({
                    "content": "\n\n".join(tool_results)
                })
                final_responses.append(tool_summary.content)
            
            if model_results:
                model_summary = await (MODEL_SUMMARY_PROMPT | self.llm).ainvoke({
                    "content": "\n\n".join(model_results)
                })
                final_responses.append(model_summary.content)
            
            final_message = AIMessage(content="\n\n".join(final_responses))
            
            return {
                **state,
                "messages": [*original_messages, final_message] if original_messages else [final_message],
                "original_messages": original_messages,
                "tool_invocations": state.get("tool_invocations", 0)
            }
            
        except Exception as e:
            logging.error(f"生成总结时出现错误: {e}")
            error_message = AIMessage(content="抱歉,生成总结时出现错误")
            return {
                **state,
                "messages": [*original_messages, error_message] if original_messages else [error_message],
                "original_messages": original_messages,
                "tool_invocations": state.get("tool_invocations", 0)
            }

    async def save_original_messages(self, state: TaskAnalyzerState) -> TaskAnalyzerState:
        """保存原始消息历史"""
        new_state = state.copy()
        new_state["original_messages"] = state["messages"]
        return new_state

    def build_graph(self) -> StateGraph:
        """构建工作流图"""
        graph_builder = StateGraph(TaskAnalyzerState)
        
        graph_builder.add_node("agent", self.agent_decision)
        graph_builder.add_node("analyze_message", self.analyze_message)
        graph_builder.add_node("process_task", self.process_task)
        graph_builder.add_node("tools", self.tool_node)
        graph_builder.add_node("save_task_result", self.save_task_result)
        graph_builder.add_node("assemble_response", self.assemble_response)
        graph_builder.add_node("save_original_messages", self.save_original_messages)
        
        graph_builder.set_entry_point("save_original_messages")
        graph_builder.add_edge("save_original_messages", "analyze_message")
        graph_builder.add_edge("analyze_message", "process_task")
        graph_builder.add_edge("process_task", "agent")
        graph_builder.add_conditional_edges("agent", self.should_continue, {"tools": "tools", "end": END})
        graph_builder.add_edge("tools", "agent")
        graph_builder.add_edge("agent", "save_task_result")
        graph_builder.add_conditional_edges(
            "save_task_result",
            self.has_more_tasks,
            {"process_next_task": "process_task", "assemble_response": "assemble_response"},
        )
        
        graph_builder.add_edge("assemble_response", END)
        return graph_builder.compile()

    async def run(self, content: str) -> Dict[str, Any]:
        """运行任务分析"""
        initial_state = {
            "messages": [HumanMessage(content=content)],
            "chat_analyzer_messages": [],
            "tasks": [],
            "current_task_index": 0,
            "task_results": {},
            "processed_tasks": set(),
            "tool_invocations": 0,
            "original_messages": []
        }
        
        try:
            result = await self.graph.ainvoke(initial_state)
            logging.info("任务分析完成")
            return result
        except Exception as e:
            logging.error(f"任务分析执行失败: {e}", exc_info=True)
            return {"error": str(e)}