"""
任务分析相关的提示词模板
"""

from langchain_core.prompts import PromptTemplate

# 任务分析提示词：用于分析用户输入中包含的独立任务
TASK_ANALYSIS_PROMPT = PromptTemplate.from_template(
    """请分析用户的输入，识别出其中包含的所有独立任务。
    
    用户输入: {user_content}
    
    请以JSON数组格式返回任务列表，每个任务包含以下字段:
    - id: 任务唯一标识符
    - content: 任务内容描述
    - requires_tool: 是否需要使用工具 (true/false)
    - tool_call: 如需使用工具，指定工具名称 (get_current_weather 或 get_current_time)

    示例输出:
    [
      {{
        "id": "task_1",
        "content": "查询北京的天气",
        "requires_tool": true,
        "tool_call": "get_current_weather"
      }}
    ]"""
) 