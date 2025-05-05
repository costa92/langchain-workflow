#! /usr/bin/env python3
from langgraph.graph import StateGraph
from llm.llm import init_ollama
from langchain.prompts import PromptTemplate
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from typing import Annotated, List
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict


system_prompt = """
# 角色与目标
你是一个翻译专家，精通各种语言的翻译，根据用户输入的内容，将内容翻译成指定用户指定的语言。

# 输出要求
1. 输入的内容，需要根据用户指定的语言进行翻译。 
2. 翻译的内容，需要符合用户指定的语言的语法和语法规则。
"""

class TranslationState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    text: str
    target_language: str
    translated_text: str

class TranslationAgent:
    def __init__(self):
        self.llm = init_ollama()    
        self.build_graph = self.build_graph()

    def translate(self, state: TranslationState) -> TranslationState:
        """
        执行翻译任务
        
        Args:
            state: 包含翻译所需信息的状态对象
            
        Returns:
            更新后的状态对象
        """
        # 构建翻译提示
        prompt = PromptTemplate.from_template(
            "请将以下文本翻译成{target_language}:\n{text}"
        )
        
        # 添加翻译请求消息
        state["messages"].append(
            HumanMessage(content=prompt.format(
                target_language=state["target_language"],
                text=state["text"]
            ))
        )
        
        # 调用 LLM 进行翻译
        response = self.llm.invoke(state["messages"])
        
        # 保存翻译结果
        state["translated_text"] = response.content
        state["messages"].append(response)
        
        return state
    def build_graph(self) -> StateGraph:
        graph = StateGraph(TranslationState)
        graph.add_node("translate", self.translate)
        graph.add_edge("__start__", "translate")
        graph.add_edge("translate", "__end__")
        return graph.compile()
    def run(self, text: str, target_language: str) -> TranslationState:
        state = TranslationState(
            text=text, 
            target_language=target_language,
            messages=[
                SystemMessage(content=system_prompt.format(text=text))
            ]
        )
        return self.build_graph.invoke(state)
  