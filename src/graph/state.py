#! /usr/bin/env python3

from typing import Annotated, Dict, Any, List
from dataclasses import dataclass
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

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "AnalyzeMessageTask":
        return cls(**json_data)


class State(TypedDict):
    """状态图的类型定义"""
    # 消息历史记录，使用add_messages函数追加新消息
    messages: Annotated[List, add_messages]
    # 待处理的任务列表
    tasks: List[AnalyzeMessageTask]
