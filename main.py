#! /usr/bin/env python3

import asyncio
from graph.generate_img import generate_img
from langchain_core.messages import HumanMessage, AIMessage
from agent.task_agent import TaskAnalyzerAgent

async def main() -> None:
    """主函数：初始化工作流并执行测试查询"""
    # 构建并可视化工作流

    taskAnalyzerAgent = TaskAnalyzerAgent()
    generate_img(taskAnalyzerAgent.build_graph(), "generated_output_task.png")
    
    content="上海在什么地方? 获取北京市的天气, 现在的时间是几点钟?"
    result = await taskAnalyzerAgent.run(content=content)
    
    # 获取最终回答
    final_messages = result.get("messages", [])
    original_messages = result.get("original_messages", [])
    
    if final_messages:
        # 打印原始消息历史
        print("\n原始消息历史:")
        for msg in original_messages:
            if isinstance(msg, HumanMessage):
                print(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"助手: {msg.content}")
                
        # 打印最终汇总回答
        print("\n最终汇总回答:")
        final_answer = final_messages[-1].content
        print(final_answer)
    else:
        print("未获取到回答")

if __name__ == "__main__":
    asyncio.run(main())
