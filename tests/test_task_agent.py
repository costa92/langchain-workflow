import pytest
from agent.task_agent import TaskAnalyzerAgent, AnalyzeMessageTask, TaskAnalyzerState
from langchain_core.messages import HumanMessage, AIMessage
from llm.llm import init_deepseek
from unittest.mock import AsyncMock, MagicMock

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
                content="上海在什么地方",
                requires_tool=False,
                tool_call=None,
                result="",
                processed=False
            ),
            AnalyzeMessageTask(
                id="task_2", 
                content="获取北京市的天气",
                requires_tool=True,
                tool_call="get_current_weather",
                result="",
                processed=False
            ),
            AnalyzeMessageTask(
                id="task_3",
                content="现在的时间是几点钟",
                requires_tool=True,
                tool_call="get_current_time",
                result="",
                processed=False
            )
        ]
        
    # 构造测试消息
        test_message = HumanMessage(
            content='上海在什么地方? 获取北京市的天气, 现在的时间是几点钟?'
        )

        # 构造初始状态
        state = TaskAnalyzerState(
            messages=[test_message],
            chat_analyzer_messages=[],
            tasks=test_tasks,
            current_task_index=0,
            task_results={},
            processed_tasks=set(),
            tool_invocations=0,
            original_messages=[test_message]
        )
        
        result = await agent.process_task(state)
        print("\n")
        print(result)
        print("\n")
        for task in result["tasks"]:
            print("\n")
            print(task)

        # state.current_task_index = 1
        # result = await agent.process_task(state)
        # print("\n")
        # print(result)
        # print("\n")
        # for task in result["tasks"]:
        #     print("\n")
        #     print(task)
        
    except ImportError as e:
        pytest.skip(f"跳过测试: {str(e)}")

# agent_decision
@pytest.mark.asyncio 
async def test_agent_decision():
    """测试agent_decision方法"""
    try:
        # 初始化代理
        agent = TaskAnalyzerAgent()

         # 创建测试任务
        test_tasks = [
            AnalyzeMessageTask(
                id="task_1",
                content="上海在什么地方",
                requires_tool=False,
                tool_call=None,
                result="",
                processed=False
            ),
            AnalyzeMessageTask(
                id="task_2", 
                content="获取北京市的天气",
                requires_tool=True,
                tool_call="get_current_weather",
                result="",
                processed=False
            ),
            AnalyzeMessageTask(
                id="task_3",
                content="现在的时间是几点钟",
                requires_tool=True,
                tool_call="get_current_time",
                result="",
                processed=False
            )
        ]
        
    # 构造测试消息
        test_message = HumanMessage(
            content='上海在什么地方? 获取北京市的天气, 现在的时间是几点钟?'
        )

        # 构造初始状态
        state = TaskAnalyzerState(
            messages=[test_message],
            chat_analyzer_messages=[],
            tasks=test_tasks,
            current_task_index=0,
            task_results={},
            processed_tasks=set(),
            tool_invocations=0,
            original_messages=[test_message]
        )
        
        result = await agent.agent_decision(state)
        print("\n")
        print(result)
        print("\n")
    except Exception as e:
        pytest.fail(f"测试失败: {str(e)}")

@pytest.mark.asyncio
async def test_save_original_messages():
    """测试save_original_messages方法,验证原始消息的保存功能"""
    try:
        # 初始化代理
        agent = TaskAnalyzerAgent()
        
        # 构造测试消息
        test_message = HumanMessage(
            content='上海在什么地方? 获取北京市的天气, 现在的时间是几点钟?'
        )
        
        # 构造初始状态
        state = TaskAnalyzerState(
            messages=[test_message],
            chat_analyzer_messages=[],
            tasks=[],
            current_task_index=0,
            task_results={},
            processed_tasks=set(),
            tool_invocations=0,
            original_messages=[]
        )

        # 执行保存操作
        result = await agent.save_original_messages(state)
        
        # 验证结果
        assert isinstance(result["original_messages"], list), "original_messages应该是列表类型"
        assert len(result["original_messages"]) == 1, "应该只保存了一条原始消息"
        assert result["original_messages"][0] == test_message, "保存的消息应与原始消息一致"
        
    except Exception as e:
        pytest.fail(f"测试失败: {str(e)}")


async def test_analyze_message():
    """测试analyze_message方法"""
    try:
        # 初始化代理
        agent = TaskAnalyzerAgent()

        # 构造测试消息
        test_message = HumanMessage(
            content='上海在什么地方? 获取北京市的天气, 现在的时间是几点钟?'
        )

        # 构造初始状态
        state = TaskAnalyzerState(
            messages=[test_message],
            chat_analyzer_messages=[],
            tasks=[],
            current_task_index=0,
            task_results={},
            processed_tasks=set(),
            tool_invocations=0,
            original_messages=[test_message]
        )

        # 执行分析操作
        result = await agent.analyze_message(state)
        # 验证结果
        assert isinstance(result["tasks"], list), "tasks应该是列表类型"

        print(result)

        for task in result["tasks"]:
            print("\n")
            print(task)
    except Exception as e:
        pytest.fail(f"测试失败: {str(e)}")
        

@pytest.mark.asyncio
async def test_init_sets_tools_and_llm():
    agent = TaskAnalyzerAgent()
    assert hasattr(agent, "llm")
    assert "get_current_weather" in agent.tool_map
    assert "get_current_time" in agent.tool_map

@pytest.mark.asyncio
async def test_save_original_messages_first_time():
    agent = TaskAnalyzerAgent()
    state = {"messages": [HumanMessage(content="hi")], "original_messages": []}
    result = await agent.save_original_messages(state)
    assert "original_messages" in result
    assert result["original_messages"] == [HumanMessage(content="hi")]

@pytest.mark.asyncio
async def test_save_original_messages_not_first_time():
    agent = TaskAnalyzerAgent()
    msg = HumanMessage(content="hi")
    state = {"messages": [msg], "original_messages": [msg]}
    result = await agent.save_original_messages(state)
    assert result == {}

@pytest.mark.asyncio
async def test_analyze_message_no_human_message():
    agent = TaskAnalyzerAgent()
    state = {"messages": [], "original_messages": [], "tasks": [], "current_task_index": 0}
    result = await agent.analyze_message(state)
    assert result["tasks"] == []
    assert result["current_task_index"] == 0

@pytest.mark.asyncio
async def test_analyze_message_invalid_json(monkeypatch):
    agent = TaskAnalyzerAgent()
    msg = HumanMessage(content="test")
    state = {"messages": [msg], "original_messages": [msg], "tasks": [], "current_task_index": 0}
    # Mock LLM to return invalid JSON
    class FakeResult:
        content = "not a json"
    agent.llm = MagicMock()
    agent.llm.ainvoke = AsyncMock(return_value=FakeResult())
    result = await agent.analyze_message(state)
    assert any("Sorry" in m.content for m in result["messages"] if isinstance(m, AIMessage))

@pytest.mark.asyncio
async def test_process_task_no_tasks():
    agent = TaskAnalyzerAgent()
    state = {"tasks": [], "current_task_index": 0, "processed_tasks": set(), "task_results": {}}
    result = await agent.process_task(state)
    assert result == {}

@pytest.mark.asyncio
async def test_process_task_already_processed():
    agent = TaskAnalyzerAgent()
    task = AnalyzeMessageTask(id="1", content="test", requires_tool=False)
    state = {
        "tasks": [task],
        "current_task_index": 0,
        "processed_tasks": {"1"},
        "task_results": {}
    }
    result = await agent.process_task(state)
    assert result["current_task_index"] == 1

@pytest.mark.asyncio
async def test_has_more_tasks_true():
    agent = TaskAnalyzerAgent()
    state = {"tasks": [1, 2], "current_task_index": 0}
    result = await agent.has_more_tasks(state)
    assert result == "process_task"

@pytest.mark.asyncio
async def test_has_more_tasks_false():
    agent = TaskAnalyzerAgent()
    state = {"tasks": [1], "current_task_index": 1}
    result = await agent.has_more_tasks(state)
    assert result == "assemble_response"

@pytest.mark.asyncio
async def test_assemble_response_no_results():
    agent = TaskAnalyzerAgent()
    state = {"task_results": {}, "original_messages": []}
    result = await agent.assemble_response(state)
    assert isinstance(result["messages"][-1], AIMessage)

@pytest.mark.asyncio
async def test_run_handles_exception(monkeypatch):
    agent = TaskAnalyzerAgent()
    agent.graph = MagicMock()
    agent.graph.ainvoke = AsyncMock(side_effect=Exception("fail"))
    result = await agent.run("test")
    assert "error" in result
    assert isinstance(result["messages"][-1], AIMessage)
        



