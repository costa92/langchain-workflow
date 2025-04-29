#! /usr/bin/env python3

import asyncio

from graph.graph import workflow_builder
from graph.generate_img import generate_img
from langchain_core.messages import HumanMessage, AIMessage
from graph.state import ChatState

async def main() -> None:
    """主函数：初始化工作流并执行测试查询"""
    # 构建并可视化工作流
    workflow = await workflow_builder()
    generate_img(workflow, "generated_output.png")
    
    # 测试复杂多任务查询
    test_message = HumanMessage(
        content="上海在什么地方? 获取北京市的天气, 现在的时间是几点钟?"
    )
    
    # 执行工作流
    result = await workflow.ainvoke(
        {"messages": [test_message]},
        config={"configurable": {"thread_id": 42}, "recursion_limit": 50},
    )

    # 提取并打印响应
    response = result.get("messages", [])[-1].content
    print("\n" + "="*50)
    print("最终响应:")
    print("="*50)
    print(response)
    print("="*50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
