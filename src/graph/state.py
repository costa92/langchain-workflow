#! /usr/bin/env python3

from typing import Annotated, Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from collections import deque


def merge_original_messages(old: Optional[List], new: Optional[List]) -> List:
    """合并原始消息列表
    
    Args:
        old: 原有消息列表
        new: 新消息列表
        
    Returns:
        合并后的消息列表
    """
    if old is None:
        return new or []
    if new is None:
        return old
    return new  # 使用新消息替换旧消息


def merge_tool_invocations(old: Optional[int], new: Optional[int]) -> int:
    """合并工具调用计数
    
    Args:
        old: 原有计数
        new: 新计数
        
    Returns:
        合并后的计数
    """
    if old is None:
        return new or 0
    if new is None:
        return old
    return new  # 使用新的计数值


def merge_task_index(old: Optional[int], new: Optional[int]) -> int:
    """合并任务索引
    
    Args:
        old: 原有索引
        new: 新索引
        
    Returns:
        合并后的索引
    """
    if old is None:
        return new or 0
    if new is None:
        return old
    return new  # 使用新的索引值


def merge_task_results(old: Optional[Dict[str, Dict[str, Any]]], 
                      new: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """合并任务结果字典
    
    Args:
        old: 原有结果字典
        new: 新结果字典
        
    Returns:
        合并后的结果字典
    """
    if old is None:
        return new or {}
    if new is None:
        return old
    return {**old, **new}  # 合并两个字典，新值覆盖旧值


def merge_processed_tasks(old: Optional[Set[str]], new: Optional[Set[str]]) -> Set[str]:
    """合并已处理任务集合
    
    Args:
        old: 原有任务集合
        new: 新任务集合
        
    Returns:
        合并后的任务集合
    """
    if old is None:
        return new or set()
    if new is None:
        return old
    return old | new  # 合并两个集合


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
    processed: bool = False  # 标记任务是否已处理
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "AnalyzeMessageTask":
        return cls(**json_data)
    
    def __str__(self) -> str:
        return f"Task({self.id}: {self.content}, {'已处理' if self.processed else '未处理'})"

    def __hash__(self) -> int:
        return hash(self.id)  # 使用任务ID作为哈希值
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnalyzeMessageTask):
            return False
        return self.id == other.id


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
    
    # 使用集合进行快速去重
    existing_tasks = {task.id: task for task in old}
    result = list(existing_tasks.values())
    
    # 只添加新的未处理任务
    for task in new:
        if task.id not in existing_tasks:
            result.append(task)
    
    # 按优先级排序
    return sorted(result, key=lambda x: (-x.priority, x.id))


class ChatState(TypedDict):
    """状态图的类型定义"""
    # 消息历史记录，使用add_messages函数追加新消息
    messages: Annotated[List, add_messages]
    # 待处理的任务列表
    tasks: Annotated[List[AnalyzeMessageTask], add_tasks]
    # 原始消息历史，用于保存处理前的消息
    original_messages: Annotated[List, merge_original_messages]
    # 工具调用次数
    tool_invocations: Annotated[int, merge_tool_invocations]
    # 当前处理的任务索引
    current_task_index: Annotated[int, merge_task_index]
    # 任务处理结果列表（使用字典提高查询效率）
    task_results: Annotated[Dict[str, Dict[str, Any]], merge_task_results]
    # 任务处理缓存，避免重复处理
    processed_tasks: Annotated[Set[str], merge_processed_tasks]
