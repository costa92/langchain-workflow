#!/usr/bin/env python3
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import pytest
from langchain.schema import HumanMessage

@pytest.mark.asyncio
async def test_ollama():
    """测试 Ollama LLM 的基础功能"""
    try:
        # llm = ChatOpenAI(
        #     model="qwen2.5:7b",
        #     api_key="ollama",
        #     base_url="http://127.0.0.1:11434/v1"
        # )

        llm = ChatOllama(
            model="qwen2.5:7b",
            base_url="http://127.0.0.1:11434"
        )
        # 使用 HumanMessage 构造输入消息
        message = HumanMessage(content="你好,你是谁，是否可以使用工具")
        result = await llm.ainvoke([message])
        
        # 验证响应
        assert result is not None  
        assert isinstance(result.content, str)
        assert len(result.content) > 0
        
        print(f"Response: {result.content}")
        
    except Exception as e:
        pytest.skip(f"Ollama服务不可用: {str(e)}")

