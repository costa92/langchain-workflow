import os

# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# # 设置模型参数
# model = ChatAnthropic(
#     model="claude-3-5-sonnet-20240620",
#     temperature=0,
#     max_tokens=1000,
#     max_retries=3,
#     api_key=os.getenv("ANTHROPIC_API_KEY"),
#     base_url="https://api.gptsapi.net",
# )


# model = ChatOpenAI(
#     model="deepseek-chat",
#     temperature=0,
#     base_url="https://api.deepseek.com/v1",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
# )

model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    temperature=0,
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
)
