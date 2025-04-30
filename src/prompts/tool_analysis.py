"""
工具分析相关的提示词模板
"""

from langchain_core.prompts import PromptTemplate

# 工具分析提示词：用于分析是否需要使用工具
TOOL_ANALYSIS_PROMPT = PromptTemplate.from_template(
    """分析用户输入是否需要使用工具:
    - 天气查询使用 get_current_weather
    - 时间查询使用 get_current_time
    
    用户输入: {input}
    
    只返回工具名称,如果不需要工具返回 "none"
    """
) 