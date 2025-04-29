#! /usr/bin/env python3

from typing import Annotated, Dict, Any, List, Optional
from dataclasses import dataclass, field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


@dataclass
class AnalyzeMessageTask:
    """任务分析结果的数据类"""
    id: str
    content: str
    type: str  
    priority: int
    requires_tool: bool
    tool_call: str
    result: str = ""
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "AnalyzeMessageTask":
        return cls(**json_data)
    
    def __str__(self) -> str:
        return f"Task({self.id}: {self.content})"


def add_tasks(old: Optional[List[AnalyzeMessageTask]], 
              new: Optional[List[AnalyzeMessageTask]]) -> List[AnalyzeMessageTask]:
    """用于 ChatState.tasks 的多步追加合并。
    
    Args:
        old: 原有任务列表
        new: 新增任务列表
        
    Returns:
        合并后的任务列表
    """
    if old is None:
        return new or []
    if new is None:
        return old
    # 使用任务ID去重，避免重复任务
    old_ids = {task.id for task in old}
    unique_new = [task for task in new if task.id not in old_ids]
    return old + unique_new


class ChatState(TypedDict):
    """状态图的类型定义"""
    # 消息历史记录，使用add_messages函数追加新消息
    messages: Annotated[List, add_messages]
    # 待处理的任务列表
    tasks: Annotated[List[AnalyzeMessageTask], add_tasks]
    # 原始消息历史，用于保存处理前的消息
    original_messages: Optional[List]
    # 工具调用次数
    tool_invocations: int
    # 当前处理的任务索引
    current_task_index: int
    # 任务处理结果列表
    task_results: List[Dict[str, Any]]
