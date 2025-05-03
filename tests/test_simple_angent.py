from agent.simple_angent import SimpleAgent
from llm.llm import init_ollama, init_deepseek
import pytest
import json

@pytest.mark.asyncio
async def test_simple_agent():
    llm = init_ollama()
    # llm = init_deepseek()
    agent = SimpleAgent(llm)
    # 用户输入的内容
    user_input = """
计算10 * 10
    """
    result_stream = await agent.run(user_input)
    async for event in result_stream:
        print("--- New Event ---")
        print(event)
        print("--- End of Event ---")
    # 验证最后一条消息是否包含有效的 JSON 响应
    final_message = event["agent"]["messages"][-1]
    # 将 JSON 字符串解析为 Python 字典
    response_dict = json.loads(final_message.content)
    print("\n")
    print("analysis_message:", response_dict["analysis_message"])
    print("user_input:", response_dict["user_input"])