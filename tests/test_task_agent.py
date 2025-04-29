import pytest
from agent.task_agent import TaskAnalyzerAgent, AnalyzeMessageTask
from langchain_core.messages import HumanMessage
from llm.llm import init_deepseek

@pytest.mark.asyncio
async def test_process_task():
    """测试process_task方法"""
    try:
        # 初始化代理
        llm = init_deepseek()
        agent = TaskAnalyzerAgent(llm)
        
        # 创建测试任务
        test_tasks = [
            AnalyzeMessageTask(
                id="task_1",
                content="获取北京的天气",
                requires_tool=True,
                tool_call="get_current_weather"
            ),
            AnalyzeMessageTask(
                id="task_2", 
                content="上海在哪里",
                requires_tool=False,
                tool_call=""
            )
        ]
        
        # 测试工具调用任务
        state = {
            "tasks": test_tasks,
            "current_task_index": 0,
            "processed_tasks": set(),
            "messages": []
        }
        
        result = await agent.process_task(state)
        assert "messages" in result, "返回结果中应包含messages字段"
        assert isinstance(result["messages"][-1], HumanMessage), "最后一条消息应为HumanMessage类型"
        assert "get_current_weather" in result["messages"][-1].content, "工具调用任务应包含工具名称"
        
        # 测试非工具调用任务
        state["current_task_index"] = 1
        result = await agent.process_task(state)
        assert "messages" in result, "返回结果中应包含messages字段"
        assert isinstance(result["messages"][-1], HumanMessage), "最后一条消息应为HumanMessage类型"
        assert result["messages"][-1].content == "上海在哪里", "非工具调用任务内容应保持不变"
        
        # 测试任务索引超出范围的情况
        state["current_task_index"] = len(test_tasks)
        result = await agent.process_task(state)
        assert result == state, "任务索引超出范围时应返回原始状态"
        
        # 测试已处理任务的情况
        state["current_task_index"] = 0
        state["processed_tasks"].add("task_1")
        result = await agent.process_task(state)
        assert result["current_task_index"] == 1, "已处理任务应跳转到下一个任务"

        print(result)
        
    except ImportError as e:
        pytest.skip(f"跳过测试: {str(e)}")

# agent_decision
@pytest.mark.asyncio 
async def test_agent_decision():
    """测试agent_decision方法"""
    try:
        # 初始化代理
        agent = TaskAnalyzerAgent()

        
        # 测试天气查询工具调用
        state = {
            "messages": [HumanMessage(content="北京今天天气怎么样?")]
        }
        result = await agent.agent_decision(state)
        print(result)
        assert "tool_name" in result, "天气查询应触发工具调用"
        assert result["tool_name"] == "get_current_weather", "应使用天气查询工具"
        

                
        # 测试常规对话
        state = {
            "messages": [HumanMessage(content="你好,请问你是谁?")]
        }
        result = await agent.agent_decision(state)
        print(result)
        assert "messages" in result, "返回结果中应包含messages字段"
        assert len(result["messages"]) > 1, "应该包含回复消息"

        # 测试时间查询工具调用
        state = {
            "messages": [HumanMessage(content="现在几点了?")]
        }
        result = await agent.agent_decision(state)
        assert "tool_name" in result, "时间查询应触发工具调用"
        assert result["tool_name"] == "get_current_time", "应使用时间查询工具"
        
        print(result)
    except Exception as e:
        pytest.fail(f"测试失败: {str(e)}")