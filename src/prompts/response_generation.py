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

FUSION_SUMMARY_PROMPT = PromptTemplate.from_template(
    """你是一个智能助手，请将以下工具调用结果和模型对话内容进行融合，总结为一段自然、连贯、面向用户的回复。

    工具结果:
    {tool_content}

    对话内容:
    {model_content}

    要求：用简洁、自然的语言，合并所有信息，避免重复，确保内容通顺且有逻辑。
    """
) 