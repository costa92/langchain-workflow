"""
响应生成相关的提示词模板
"""

from langchain_core.prompts import PromptTemplate

# 工具结果总结提示词：用于总结工具调用结果
TOOL_SUMMARY_PROMPT = PromptTemplate.from_template(
    """请总结以下工具调用结果:
    {content}
    
    请生成简洁清晰的总结。"""
)

# 模型结果总结提示词：用于总结对话内容
MODEL_SUMMARY_PROMPT = PromptTemplate.from_template(
    """请总结以下对话内容:
    {content}
    
    请生成连贯的总结。"""
) 