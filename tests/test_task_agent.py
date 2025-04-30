import pytest
from agent.task_agent import TaskAnalyzerAgent, AnalyzeMessageTask, TaskAnalyzerState
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
        



