#! /usr/bin/env python3

from langgraph.graph import StateGraph
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from typing import Annotated, List
from llm.llm import init_ollama
from typing_extensions import TypedDict

system_prompt = """
# 角色与目标
你是一个语音合成专家，根据用户输入的内容，将内容转换为语音，并保存为音频文件。

# 输出要求

"""
class TextSpeechState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    text: str
    audio_file: str

class TextSpeechAgent:
    def __init__(self):
        self.llm = init_ollama()

    def build_graph(self) -> StateGraph:
        graph = StateGraph(TextSpeechState)
        graph.add_node("text_speech", self.text_speech)
        graph.add_edge("start", "text_speech")
        graph.add_edge("text_speech", "end")
        return graph.compile()

    def text_speech(self, state: TextSpeechState) -> TextSpeechState:
        """
        执行语音合成任务
        
        Args:
            state: 包含语音合成所需信息的状态对象
            
        Returns:
            更新后的状态对象
        """
        # 构建语音合成提示
        prompt = PromptTemplate.from_template(
            "请将以下文本转换为语音:\n{text}"
        )
        
        # 添加语音合成请求消息
        state["messages"].append(
            HumanMessage(content=prompt.format(text=state["text"]))
        )

        # 调用 LLM 进行语音合成
        response = self.llm.invoke(state["messages"])

        # 保存语音合成结果
        state["audio_file"] = response.content
        state["messages"].append(response)
        
        return state


    def run(self, text: str) -> TextSpeechState:
        state = TextSpeechState(
            text=text,
            messages=[
                SystemMessage(content=system_prompt.format(text=text))
            ]
        )
        return self.build_graph.invoke(state)