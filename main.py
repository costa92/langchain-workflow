#! /usr/bin/env python3

import asyncio

from graph.graph import parallelWorkflow
from langchain_core.messages import HumanMessage, AIMessage


async def main() -> None:
    # 使用更复杂的指令来测试多任务处理能力
    result = await parallelWorkflow.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="上海在什么地方? 获取上海市的天气, 现在的时间是几点钟?"
                ),
            ]
        },
        config={"configurable": {"thread_id": 42}, "recursion_limit": 50},
    )

    # 打印完整的消息内容
    print("最终响应:")
    print(result.get("messages")[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
