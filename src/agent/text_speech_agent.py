#! /usr/bin/env python3

import torch
import ChatTTS
from langgraph.graph import StateGraph
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from typing import Annotated, List, Optional
from llm.llm import init_ollama
from typing_extensions import TypedDict
from langchain_core.prompts import PromptTemplate

system_prompt = """
# 角色与目标
你是一个语音合成专家，根据用户输入的内容，将内容转换为语音，并保存为音频文件。

# 输出要求
- 生成自然流畅的语音
- 支持中英文混合内容
- 语音语调自然，富有表现力
"""

class TextSpeechState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    text: str
    audio_file: str
    speaker_embedding: Optional[torch.Tensor]

class TextSpeechAgent:
    def __init__(self):
        self.llm = init_ollama()
        self.chat_tts = ChatTTS.Chat()
        self.chat_tts.load(compile=False)
        
        # 加载说话人统计信息
        try:
            self.spk_stat = torch.load('asset/spk_stat.pt')
        except FileNotFoundError:
            raise RuntimeError("Speaker stats file not found at asset/spk_stat.pt")

    def build_graph(self) -> StateGraph:
        graph = StateGraph(TextSpeechState)
        graph.add_node("text_speech", self.text_speech)
        graph.set_entry_point("text_speech")  # 设置入口点而不是使用"start"
        graph.add_edge("text_speech", "end")
        return graph.compile()

    def generate_speaker_embedding(self) -> torch.Tensor:
        """生成随机说话人嵌入"""
        return torch.randn(768) * self.spk_stat.chunk(2)[0] + self.spk_stat.chunk(2)[1]

    def text_speech(self, state: TextSpeechState) -> TextSpeechState:
        """
        执行语音合成任务
        
        Args:
            state: 包含语音合成所需信息的状态对象
            
        Returns:
            更新后的状态对象
        """
        # 生成说话人嵌入
        if state.get("speaker_embedding") is None:
            state["speaker_embedding"] = self.generate_speaker_embedding()
            
        # 设置合成参数
        params_infer_code = {
            'spk_emb': state["speaker_embedding"],
            'temperature': 0.3
        }
        params_refine_text = {
            'prompt': '[oral_2][laugh_0][break_6]'
        }

        # 执行语音合成
        try:
            wav = self.chat_tts.infer(
                state["text"],
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code
            )
            output_file = f"output_{hash(state['text'])}.wav"
            torch.save(wav[0], output_file)
            state["audio_file"] = output_file
            state["messages"].append(
                AIMessage(content=f"语音已成功合成并保存至 {output_file}")
            )
        except Exception as e:
            state["messages"].append(
                AIMessage(content=f"语音合成失败: {str(e)}")
            )
            
        return state

    def run(self, text: str) -> TextSpeechState:
        """
        运行语音合成流程
        
        Args:
            text: 需要合成的文本内容
            
        Returns:
            包含合成结果的状态对象
        """
        state = TextSpeechState(
            text=text,
            messages=[
                SystemMessage(content=system_prompt)
            ],
            audio_file="",
            speaker_embedding=None
        )
        return self.build_graph().invoke(state)

if __name__ == "__main__":
    agent = TextSpeechAgent()
    result = agent.run("用中英文混杂的方式，简单介绍宁波")  # 直接传入字符串而不是字典
    print(f"Audio saved to: {result['audio_file']}")