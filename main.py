#! /usr/bin/env python3

import asyncio

from graph.graph import workflow_builder, generate_img
from langchain_core.messages import HumanMessage, AIMessage
from graph.state import State

async def main() -> None:
    
    state = State(
        messages=[],
        tasks=[],
        tool_invocations=0,
        current_task_index=0,
        task_results=[]
    )
    parallelWorkflow = await workflow_builder(state)

    generate_img(parallelWorkflow, "generated_output.png")
    # 使用更复杂的指令来测试多任务处理能力
    result = await parallelWorkflow.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="上海在什么地方? 获取北京市的天气, 现在的时间是几点钟?"
                ),
            ]
        },
        config={"configurable": {"thread_id": 42}, "recursion_limit": 50},
    )

    # 打印完整的消息内容
    print("最终响应:start","\n\n")
    print(result.get("messages")[-1].content)
    print("最终响应:end","\n\n")



if __name__ == "__main__":
    asyncio.run(main())
