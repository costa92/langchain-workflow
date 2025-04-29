from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Any, Dict
import logging

from graph.graph import workflow_builder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from graph.state import ChatState

app = FastAPI(title="Langchain Workflow Chat API")

# 使用langchain_core的消息类型，而不是自定义Message类
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., description="消息历史，最后一条为用户输入")

class ChatResponse(BaseModel):
    """聊天响应模型"""
    reply: str = Field(..., description="AI回复的内容")
    messages: List[Dict[str, str]] = Field(..., description="完整的消息历史记录")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # 将请求中的消息转换为langchain_core的消息类型
        langchain_messages = []
        for msg in request.messages:
            if msg["role"] == "human":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                langchain_messages.append(AIMessage(content=msg["content"]))
    
        # 构建工作流
        parallelWorkflow = await workflow_builder()
        
        # 调用工作流处理
        result = await parallelWorkflow.ainvoke(
            {"messages": langchain_messages},
            config={"configurable": {"thread_id": 42}, "recursion_limit": 50},
        )
        
        messages = result.get("messages", [])
        reply = messages[-1].content if messages else ""
        
        # 将消息转换为可哈希的字典类型
        formatted_messages = [{"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content} for m in messages]
        
        return ChatResponse(reply=reply, messages=formatted_messages)
    except Exception as e:
        logging.error(f"API 调用出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="模型推理失败")
