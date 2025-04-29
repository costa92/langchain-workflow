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
from langchain_core.prompts import PromptTemplate
import json
from typing_extensions import TypedDict
from dataclasses import dataclass


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

def merge_tasks(old, new):
    if old is None:
        return new or []
    if new is None:
        return old
    old_ids = {t.id for t in old}
    return old + [t for t in new if t.id not in old_ids]

def merge_task_index(old, new):
    if old is None:
        return new or 0
    if new is None:
        return old
    return new

def merge_task_results(old, new):
    if old is None:
        return new or {}
    if new is None:
        return old
    return {**old, **new}

def merge_processed_tasks(old, new):
    if old is None:
        return new or set()
    if new is None:
        return old
    return old | new

def merge_tool_invocations(old, new):
    if old is None:
        return new or 0
    if new is None:
        return old
    return new

def merge_original_messages(old, new):
    if old is None:
        return new or []
    if new is None:
        return old
    return new

class TaskAnalyzerState(TypedDict):
    """任务分析器状态类型定义"""
    messages: Annotated[List[AnyMessage], add_messages]
    chat_analyzer_messages: Annotated[List[AnyMessage], add_messages]
    tasks: Annotated[List[AnalyzeMessageTask], merge_tasks]
    current_task_index: Annotated[int, merge_task_index]
    task_results: Annotated[Dict[str, Any], merge_task_results]
    processed_tasks: Annotated[set, merge_processed_tasks]
    tool_invocations: Annotated[int, merge_tool_invocations]
    original_messages: Annotated[List[AnyMessage], merge_original_messages]


class TaskAnalyzerAgent:
    """
    任务分析代理类，负责分析用户输入的任务并提供相应的响应
    """
    def __init__(self, llm: BaseChatModel = None):
        """
        初始化任务分析代理
        
        参数:
            llm: 大语言模型实例，如果为None则使用默认的deepseek模型
        """
        # 如果没有提供LLM，则默认使用deepseek
        self.llm = llm if llm is not None else init_deepseek()

        # 定义并绑定工具
        self.tools = [get_current_weather, get_current_time]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # 初始化工具节点
        self.tool_node = ToolNode(self.tools)
        
        # 构建工作流图
        self.graph = self.build_graph()
        
        logging.info("任务分析代理初始化完成")

    async def agent_decision(self, state: TaskAnalyzerState) -> Dict[str, Any]:
        """
        代理决策函数，处理当前状态并返回下一步操作
        
        参数:
            state: 当前任务分析器状态
            
        返回:
            包含下一步操作的字典
        """
        try:
            messages = state.get("messages", [])
            if not messages:
                logging.warning("没有消息可处理")
                return {}
                
            # 处理消息逻辑
            logging.info(f"处理消息: {messages[-1].content if messages else 'None'}")
            
            # 检查是否需要工具调用
            last_message = messages[-1]
            if isinstance(last_message, HumanMessage):
                # 分析是否需要工具调用
                tool_analysis_prompt = PromptTemplate.from_template(
                    """分析用户输入是否需要使用工具:
                    - 天气查询使用 get_current_weather
                    - 时间查询使用 get_current_time
                    
                    用户输入: {input}
                    
                    只返回工具名称,如果不需要工具返回 "none"
                    """
                )
                tool_result = await (tool_analysis_prompt | self.llm).ainvoke({"input": last_message.content})
                tool_name = tool_result.content.strip().lower()
                
                if tool_name in ["get_current_weather", "get_current_time"]:
                    # 需要工具调用,返回工具调用信息
                    return {
                        "tool_name": tool_name,
                        "tool_input": last_message.content
                    }
                else:
                    return {"tool_name": None, "tool_input": last_message.content}  
            else:
                return {"tool_name": None, "tool_input": last_message.content}
        except Exception as e:
            logging.error(f"代理决策过程出错: {e}", exc_info=True)
            return {}

    async def analyze_message(self, state: TaskAnalyzerState) -> TaskAnalyzerState:
        """分析用户消息中的多个任务，并结构化提取出这些任务"""
        # 如果已经有任务在处理，直接返回
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
            # 调用模型分析任务
            task_analysis_prompt = PromptTemplate.from_template(
                """请分析用户的输入，识别出其中包含的所有独立任务。
                
                用户输入: {user_content}
                
                请以JSON数组格式返回任务列表，每个任务包含以下字段:
                - id: 任务唯一标识符
                - content: 任务内容描述
                - requires_tool: 是否需要使用工具 (true/false)
                - tool_call: 如需使用工具，指定工具名称 (get_current_weather 或 get_current_time)

                示例输出:
                [
                  {{
                    "id": "task_1",
                    "content": "查询北京的天气",
                    "requires_tool": true,
                    "tool_call": "get_current_weather"
                  }}
                ]"""
            )
            
            result = await (task_analysis_prompt | self.llm).ainvoke({"user_content": last_message.content})
            
            # 清理和解析JSON
            json_str = result.content.strip()
            start, end = json_str.find('['), json_str.rfind(']') + 1
            if start != -1 and end > 0:
                json_str = json_str[start:end]
                
            data = json.loads(json_str)
            new_tasks = [AnalyzeMessageTask.from_json(task_data) for task_data in data]
            
            # 过滤掉已处理的任务
            new_tasks = [task for task in new_tasks if task.id not in processed_tasks]
            
            if not new_tasks:
                logging.info("没有新的待处理任务")
                return state
                
            logging.info(f"分析出 {len(new_tasks)} 个新任务")
            for task in new_tasks:
                logging.info(f"- {task}")
            
            # 重置任务相关状态
            return {
                **state,
                "tasks": new_tasks,
                "current_task_index": 0,
                "task_results": {},  # 使用字典存储结果
                "processed_tasks": processed_tasks  # 保持已处理任务的记录
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
        
        # 如果任务已处理，跳过
        if current_task.id in processed_tasks:
            return {
                **state,
                "current_task_index": current_index + 1
            }
        
        logging.info(f"处理任务 {current_index + 1}/{len(tasks)}: {current_task}")

        # 处理工具调用
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
            # 如果不需要工具调用,直接返回结果
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

        print("task_results",task_results)

        if not tasks or current_index >= len(tasks):
            return state

        current_task = tasks[current_index]
        
        # 分别收集模型回答和工具调用结果
        ai_responses = []
        tool_responses = []
        
        for msg in messages[-3:]:
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                ai_responses.append(msg.content)
            elif isinstance(msg, ToolMessage):
                tool_responses.append(f"工具【{getattr(msg, 'name', '未知工具')}】执行结果: {msg.content}")

        # 根据任务类型选择结果
        if current_task.requires_tool:
            result = "\n".join(tool_responses) if tool_responses else "工具调用无结果"
        else:
            result = "\n".join(ai_responses) if ai_responses else "模型回答无结果"

        # 更新任务结果
        if not current_task.result:
            current_task.result = result
        
        # 更新处理状态
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

        print("task_results", task_results)

        if not task_results:
            return state

        # 分别收集工具结果和模型回答
        tool_results = []
        model_results = []
        
        for result in task_results.values():
            if result['task'] and result['task'].result:
                if result.get('is_tool_result'):
                    tool_results.append(result['task'].result)
                else:
                    model_results.append(result['task'].result)

        try:
            # 分别生成工具结果和模型回答的总结
            final_responses = []
            
            if tool_results:
                tool_prompt = PromptTemplate.from_template(
                    """请总结以下工具调用结果:
                    {content}
                    
                    请生成简洁清晰的总结。"""
                )
                tool_summary = await (tool_prompt | self.llm).ainvoke({"content": "\n\n".join(tool_results)})
                final_responses.append(tool_summary.content)
            
            if model_results:
                model_prompt = PromptTemplate.from_template(
                    """请总结以下对话内容:
                    {content}
                    
                    请生成连贯的总结。"""
                )
                model_summary = await (model_prompt | self.llm).ainvoke({"content": "\n\n".join(model_results)})
                final_responses.append(model_summary.content)
            
            # 组合最终回复
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
        """
        构建工作流图
        
        返回:
            配置好的状态图实例
        """
        # 构建工作流图
        graph_builder = StateGraph(TaskAnalyzerState)
        
        # 添加节点
        graph_builder.add_node("agent", self.agent_decision)  # 代理决策节点，处理当前状态并返回下一步操作
        graph_builder.add_node("analyze_message", self.analyze_message)  # 消息分析节点，分析用户消息中的多个任务
        graph_builder.add_node("process_task", self.process_task)  # 任务处理节点，处理当前任务并更新状态
        graph_builder.add_node("tools", self.tool_node)  # 工具节点，执行工具调用操作
        graph_builder.add_node("save_task_result", self.save_task_result)  # 保存任务结果节点，存储任务执行结果
        graph_builder.add_node("assemble_response", self.assemble_response)  # 组装响应节点，汇总所有任务结果生成最终回复
        graph_builder.add_node("save_original_messages", self.save_original_messages)  # 保存原始消息节点，备份初始消息历史
        
        # 设置图的边和流程
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
        # 编译工作流图
        return graph_builder.compile()

    async def run(self, content: str) -> Dict[str, Any]:
        """
        运行任务分析
        
        参数:
            task: 用户输入的任务文本
            
        返回:
            任务分析结果
        """
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
            # 执行工作流
            result = await self.graph.ainvoke(initial_state)
            logging.info("任务分析完成")
            return result
        except Exception as e:
            logging.error(f"任务分析执行失败: {e}", exc_info=True)
            return {"error": str(e)}