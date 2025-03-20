#! /usr/bin/env python3

import asyncio

from graph.graph import parallelWorkflow
from langchain_core.messages import HumanMessage,AIMessage


async def main() -> None:
   # 使用更明确的指令来调用工具
   result = await parallelWorkflow.ainvoke(
       {"messages": [
           HumanMessage(content="上海在什么地方?,获取上海市的天气"),
       ]},
      config={"configurable": {"thread_id": 42}}
   )
   
   # 打印完整的消息内容
   print("最终响应:")
   print(result.get("messages")[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
