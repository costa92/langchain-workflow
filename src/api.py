from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Any, Dict
import logging

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from agent.task_agent import TaskAnalyzerAgent
app = FastAPI(title="Langchain Workflow Chat API")


@app.get("/health")
async def health():
    """检查接口接口"""
    return "OK"


# 使用langchain_core的消息类型，而不是自定义Message类
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., description="消息历史，最后一条为用户输入")

class ChatResponse(BaseModel):
    """聊天响应模型"""
    reply: str = Field(..., description="AI回复的内容")
    messages: List[Dict[str, str]] = Field(..., description="完整的消息历史记录")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """处理聊天请求的端点
    
    Args:
        request: 包含消息历史的请求对象
        
    Returns:
        ChatResponse: 包含AI回复和完整消息历史的响应
        
    Raises:
        HTTPException: 当模型推理失败时抛出500错误
    """
    try:
        # 将请求消息转换为langchain格式
        langchain_messages = [
            HumanMessage(content=msg["content"]) if msg["role"] == "human"
            else AIMessage(content=msg["content"])
            for msg in request.messages
            if msg["role"] in ("human", "ai")
        ]

        # 初始化任务分析代理并执行工作流
        taskAnalyzerAgent = TaskAnalyzerAgent()
        result = await taskAnalyzerAgent.run( 
            content=request.messages[-1]["content"]
        )
        
        # 提取结果消息
        messages = result.get("messages", [])
        if not messages:
            raise ValueError("模型未返回任何消息")
            
        # 格式化响应
        formatted_messages = [
            {
                "role": "human" if isinstance(m, HumanMessage) else "ai",
                "content": m.content
            } 
            for m in messages
        ]
        
        return ChatResponse(
            reply=messages[-1].content,
            messages=formatted_messages
        )
        
    except Exception as e:
        logging.error(f"聊天API调用失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"模型推理失败: {str(e)}"
        )

