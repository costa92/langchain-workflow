#! /usr/bin/env python3
import logging
from typing import Annotated, List, Dict, Any, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from llm.llm import init_deepseek  # Assuming this initializes your LLM
from tools.tools import get_current_weather, get_current_time # Assuming these are your tool functions
# from langgraph.prebuilt import ToolNode # No longer needed
import json
from typing_extensions import TypedDict
from dataclasses import dataclass, field
from langchain.prompts import PromptTemplate

# Import prompt templates (assuming these exist in the specified paths)
from prompts.task_analysis import TASK_ANALYSIS_PROMPT
# from prompts.tool_analysis import TOOL_ANALYSIS_PROMPT # Potentially not needed if analysis is done elsewhere
from prompts.response_generation import TOOL_SUMMARY_PROMPT, MODEL_SUMMARY_PROMPT, FUSION_SUMMARY_PROMPT

# --- Data Classes and Merge Functions (Mostly unchanged) ---

@dataclass
class AnalyzeMessageTask:
    """任务分析结果的数据类"""
    id: str
    content: str
    requires_tool: bool
    tool_call: str = "" # Default to empty string if not provided
    result: str = ""
    processed: bool = False  # 标记任务是否已处理

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "AnalyzeMessageTask":
        # Ensure tool_call exists, provide default if missing
        json_data['tool_call'] = json_data.get('tool_call', '')
        return cls(**json_data)

def merge_tasks(old: List[AnalyzeMessageTask], new: List[AnalyzeMessageTask]) -> List[AnalyzeMessageTask]:
    if old is None:
        return new or []
    if new is None:
        return old
    # More robust merging: update existing tasks if re-analyzed, add new ones
    merged_dict = {t.id: t for t in old}
    for task in new:
        if task.id in merged_dict:
            # Optionally update fields if needed, e.g., overwrite result/processed status
            # merged_dict[task.id] = task # Example: Overwrite completely
            pass # Or just keep the old one if no update logic is needed
        else:
            merged_dict[task.id] = task
    return list(merged_dict.values())


def merge_task_index(old: int, new: int) -> int:
    # Reset index when new tasks arrive
    return new if new is not None else old or 0

def merge_task_results(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    if old is None: return new or {}
    if new is None: return old
    # Simple merge, new keys overwrite old ones if necessary
    return {**old, **new}

def merge_processed_tasks(old: set, new: set) -> set:
    if old is None: return new or set()
    if new is None: return old
    return old | new

# tool_invocations might not be needed if we don't limit calls this way
# def merge_tool_invocations(old: int, new: int) -> int:
#     return new if new is not None else old or 0

def merge_original_messages(old: List[AnyMessage], new: List[AnyMessage]) -> List[AnyMessage]:
    # Keep the *initial* original messages, don't overwrite with intermediate ones
    return old if old else new or []

class TaskAnalyzerState(TypedDict):
    """任务分析器状态类型定义"""
    messages: Annotated[List[AnyMessage], add_messages] # Main message history for final output
    # chat_analyzer_messages might not be needed if analysis is self-contained
    tasks: Annotated[List[AnalyzeMessageTask], merge_tasks]  # All identified tasks
    current_task_index: Annotated[int, merge_task_index]  # Index of the next task to process
    task_results: Annotated[Dict[str, Any], merge_task_results]  # Stores results keyed by task ID
    processed_tasks: Annotated[set, merge_processed_tasks]  # Set of processed task IDs
    # tool_invocations: Annotated[int, merge_tool_invocations] # Optional: keep if needed for limits
    original_messages: Annotated[List[AnyMessage], merge_original_messages] # Store the very first input

# --- Task Analyzer Agent ---

class TaskAnalyzerAgent:
    """任务分析代理类 (Optimized Graph)"""

    def __init__(self, llm: BaseChatModel = None):
        """初始化任务分析代理"""
        self.llm = llm or init_deepseek()
        # Keep tools available for direct invocation
        self.tools = [get_current_weather, get_current_time]
        self.tool_map = {tool.name: tool for tool in self.tools} # Map name to function
        # self.llm_with_tools = self.llm.bind_tools(self.tools) # Not needed for this graph structure
        # self.tool_node = ToolNode(self.tools) # Removed
        self.graph = self.build_graph()
        logging.info("任务分析代理初始化完成 (Optimized Graph)")

    # Removed agent_decision, should_continue, save_task_result as their logic is integrated or removed

    async def save_original_messages(self, state: TaskAnalyzerState) -> Dict[str, Any]:
        """保存原始消息历史 (Runs once at the start)"""
        # Ensure original_messages is only set once
        if not state.get("original_messages"):
             logging.info("Saving original messages.")
             return {"original_messages": state["messages"]}
        return {} # No change needed if already set

    async def analyze_message(self, state: TaskAnalyzerState) -> Dict[str, Any]:
        """分析用户消息中的多个任务,并结构化提取出这些任务"""
        # Only analyze if no tasks exist or all previous tasks are processed
        # This prevents re-analysis if we loop back for some reason.
        tasks = state.get("tasks", [])
        current_index = state.get("current_task_index", 0)
        if tasks and current_index < len(tasks):
             logging.debug("Tasks already exist, skipping analysis.")
             return {} # No change needed


        messages = state["messages"]
        # Use original_messages if available, otherwise the current messages list
        source_messages = state.get("original_messages") or messages
        last_human_message = next((msg for msg in reversed(source_messages) if isinstance(msg, HumanMessage)), None)

        if not last_human_message:
            logging.warning("未找到用户消息 for analysis.")
            # If no human message, maybe end the process? Or handle differently.
            return {"tasks": [], "current_task_index": 0} # Ensure state is consistent

        logging.info(f"开始分析新消息: {last_human_message.content}")
        try:
            # Use the LLM to extract tasks based on the prompt
            analysis_result = await (TASK_ANALYSIS_PROMPT | self.llm).ainvoke({
                "user_content": last_human_message.content
            })

            logging.debug(f"LLM Analysis raw output: {analysis_result.content}")

            # Robust JSON extraction
            json_str = analysis_result.content
            try:
                # Try direct parsing first
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # If direct parse fails, try finding the list brackets
                start = json_str.find('[')
                end = json_str.rfind(']') + 1
                if start != -1 and end > 0:
                    json_str_extracted = json_str[start:end]
                    try:
                        data = json.loads(json_str_extracted)
                    except json.JSONDecodeError as e_inner:
                        logging.error(f"Failed to parse JSON even after extraction: {json_str_extracted}. Error: {e_inner}")
                        # Handle error - maybe add a message to the user or end graph
                        return {"messages": state["messages"] + [AIMessage(content="Sorry, I couldn't understand the tasks in your request.")]}
                else:
                    logging.error(f"Could not find JSON list in the analysis output: {json_str}")
                    # Handle error
                    return {"messages": state["messages"] + [AIMessage(content="Sorry, I had trouble structuring the tasks.")]}


            # Ensure data is a list
            if not isinstance(data, list):
                 logging.error(f"Analysis result is not a list: {data}")
                 return {"messages": state["messages"] + [AIMessage(content="Sorry, the task analysis result was not in the expected format.")]}


            new_tasks = [AnalyzeMessageTask.from_json(task_data) for task_data in data]
            logging.info(f"分析出 {len(new_tasks)} 个新任务")
            for task in new_tasks:
                logging.info(f"- {task}")

            # Return new tasks and reset index
            return {
                "tasks": new_tasks,
                "current_task_index": 0,
                "task_results": {}, # Reset results for new analysis
                "processed_tasks": set() # Reset processed set
            }

        except Exception as e:
            logging.error(f"任务分析错误: {e}", exc_info=True)
            # Return state with an error message maybe?
            error_message = AIMessage(content=f"An error occurred during task analysis: {e}")
            return {"messages": state["messages"] + [error_message]} # Add error to history

    async def process_task(self, state: TaskAnalyzerState) -> Dict[str, Any]:
        """
        处理当前任务: 如果需要工具,直接调用; 否则,调用LLM获取结果.
        然后保存结果并更新状态.
        """
        tasks = state.get("tasks", [])
        current_index = state.get("current_task_index", 0)
        processed_tasks = state.get("processed_tasks", set())
        task_results = state.get("task_results", {})

        if not tasks or current_index >= len(tasks):
            logging.info("No more tasks to process or task list empty.")
            # This should ideally not be reached if has_more_tasks logic is correct
            # but good as a safeguard.
            return {} # No changes needed

        current_task = tasks[current_index]

        # Skip if already processed (e.g., if graph logic had an unexpected loop)
        if current_task.id in processed_tasks:
            logging.warning(f"Task {current_task.id} already processed, skipping. Moving to next.")
            return {"current_task_index": current_index + 1}

        logging.info(f"Processing task {current_index + 1}/{len(tasks)} (ID: {current_task.id}): {current_task.content}")

        task_result_content = None
        is_tool_result = False

        try:
            if current_task.requires_tool and current_task.tool_call in self.tool_map:
                tool_func = self.tool_map[current_task.tool_call]
                logging.info(f"Task requires tool: {current_task.tool_call}")
                is_tool_result = True
                tool_input = {} # Default empty dict for tools like get_current_time

                # --- Adapt Input based on Tool ---
                # This part needs specific logic per tool based on how AnalyzeMessageTask extracts info
                if current_task.tool_call == "get_current_weather":
                    # Example: Assume the task content *is* the location or extract it
                    # Option 1: Use the whole content if it's just the location
                    # tool_input = {"location": current_task.content}
                    # Option 2: Extract last word (less robust)
                    parts = current_task.content.split()
                    location = parts[-1] if parts else "Unknown" # Handle empty content
                    tool_input = {"location": location} # Match the expected input key of the tool
                    logging.info(f"Extracted location for weather: {location}")
                elif current_task.tool_call == "get_current_time":
                    # Likely needs no input, or maybe timezone? Assume no input needed.
                    tool_input = {}
                # Add more elif blocks for other tools and their specific input needs

                logging.info(f"Invoking tool {current_task.tool_call} with input: {tool_input}")
                # Use ainvoke if the tool supports async, otherwise invoke
                if hasattr(tool_func, 'ainvoke'):
                    tool_output = await tool_func.ainvoke(input=tool_input)
                else:
                    tool_output = tool_func.invoke(input=tool_input) # Fallback to sync if needed

                task_result_content = str(tool_output) # Ensure result is string
                logging.info(f"Tool {current_task.tool_call} executed successfully. Result: {task_result_content}")

            elif current_task.requires_tool:
                # Tool required but not found or specified incorrectly
                logging.warning(f"Task {current_task.id} requires tool '{current_task.tool_call}', but it's not available.")
                task_result_content = f"Error: Tool '{current_task.tool_call}' not recognized or available."
                is_tool_result = False # Treat as an error message, not a tool result

            else:
                # Task does not require a tool, use LLM to generate response
                logging.info(f"Task does not require tool. Using LLM for: {current_task.content}")
                is_tool_result = False
                # Create a simple prompt or just use the content directly
                llm_response = await self.llm.ainvoke([HumanMessage(content=current_task.content)])
                task_result_content = llm_response.content
                logging.info(f"LLM generated response for task {current_task.id}: {task_result_content}")

        except Exception as e:
            error_msg = f"Error processing task {current_task.id} ('{current_task.content}'): {str(e)}"
            logging.error(error_msg, exc_info=True)
            task_result_content = error_msg # Store the error as the result
            is_tool_result = False # Mark as not a successful tool result

        # --- Update State ---
        # Store the result associated with the task ID
        new_task_results = {
            **task_results,
            current_task.id: {
                 # Keep original task info for context if needed in assembly
                "task_content": current_task.content,
                "result": task_result_content,
                "is_tool_result": is_tool_result
            }
        }
        # Mark task as processed
        new_processed_tasks = processed_tasks | {current_task.id}
        # Increment index for the next task
        next_index = current_index + 1

        return {
            "task_results": new_task_results,
            "processed_tasks": new_processed_tasks,
            "current_task_index": next_index,
            # Optionally update messages here if we want intermediate steps logged
            # "messages": state["messages"] + [AIMessage(content=f"Processed task {current_task.id}. Result: {task_result_content}")]
        }


    async def has_more_tasks(self, state: TaskAnalyzerState) -> Literal["process_task", "assemble_response"]:
        """判断是否有更多任务需要处理"""
        tasks = state.get("tasks", [])
        current_index = state.get("current_task_index", 0)
        total_tasks = len(tasks)

        if current_index < total_tasks:
            logging.info(f"More tasks remaining ({current_index + 1} of {total_tasks}). Continuing to process_task.")
            return "process_task"
        else:
            logging.info("All tasks processed. Proceeding to assemble_response.")
            return "assemble_response"


    async def assemble_response(self, state: TaskAnalyzerState) -> Dict[str, Any]:
        """组装所有任务处理结果为最终响应"""
        task_results = state.get("task_results", {})

        logging.info(f"组装所有任务处理结果为最终响应: {task_results}")
        original_messages = state.get("original_messages", []) # Use the initial messages

        if not task_results:
            logging.warning("No task results found to assemble response.")
            # Handle case with no results - maybe return original message or a default response
            final_message = AIMessage(content="I couldn't find any specific tasks to address in your request.")
            return {"messages": original_messages + [final_message]}


        tool_results_content = []
        model_results_content = []

        # Iterate through the collected results
        # Ensure consistent order if tasks list is available, otherwise just iterate results dict
        tasks = state.get("tasks", [])
        task_order = [task.id for task in tasks] if tasks else task_results.keys()

        for task_id in task_order:
             result_data = task_results.get(task_id)
             if result_data and result_data.get("result") is not None: # Check if result exists
                 result_content = result_data["result"]
                 # Add context if needed:
                 # result_prefix = f"Regarding '{result_data.get('task_content', 'your request')}':\n"
                 if result_data.get('is_tool_result'):
                     tool_results_content.append(result_content)
                 else:
                     model_results_content.append(result_content)


        final_response_content = ""
        try:
            # Use summary prompts based on what results are available
            if tool_results_content and model_results_content:
                logging.info("Assembling response using FUSION_SUMMARY_PROMPT")
                fusion_summary = await (FUSION_SUMMARY_PROMPT | self.llm).ainvoke({
                    "tool_content": "\n\n".join(tool_results_content),
                    "model_content": "\n\n".join(model_results_content)
                })
                final_response_content = fusion_summary.content
            elif tool_results_content:
                logging.info("Assembling response using TOOL_SUMMARY_PROMPT")
                # If only one tool result, maybe just return it directly? Or always summarize.
                if len(tool_results_content) == 1 and len(tasks) == 1:
                     final_response_content = tool_results_content[0]
                else:
                    tool_summary = await (TOOL_SUMMARY_PROMPT | self.llm).ainvoke({
                        "content": "\n\n".join(tool_results_content)
                    })
                    final_response_content = tool_summary.content
            elif model_results_content:
                logging.info("Assembling response using MODEL_SUMMARY_PROMPT")
                 # If only one model result, maybe just return it directly?
                if len(model_results_content) == 1 and len(tasks) == 1:
                     final_response_content = model_results_content[0]
                else:
                    model_summary = await (MODEL_SUMMARY_PROMPT | self.llm).ainvoke({
                        "content": "\n\n".join(model_results_content)
                    })
                    final_response_content = model_summary.content
            else:
                # Should be caught by the initial check, but as a fallback
                logging.warning("No valid results content found to assemble response.")
                final_response_content = "I processed your request, but there were no specific results to report."

            logging.info(f"Final assembled response: {final_response_content}")
            final_message = AIMessage(content=final_response_content.strip())
            logging.info(f"Final assembled response: {final_message.content}")

            # Return final state with original input + final response
            return {
                "messages": original_messages + [final_message]
                # Reset other fields if this is the absolute end state
                # "tasks": [],
                # "current_task_index": 0,
                # "task_results": {},
                # "processed_tasks": set()
            }

        except Exception as e:
            logging.error(f"生成总结时出现错误: {e}", exc_info=True)
            error_message = AIMessage(content="Sorry, an error occurred while assembling the final response.")
            # Return state with error message appended
            return {
                 "messages": original_messages + [error_message]
            }


    def build_graph(self) -> StateGraph:
        """构建优化的工作流图 (No ToolNode, direct calls in process_task)"""
        graph_builder = StateGraph(TaskAnalyzerState)

        # Define Nodes
        graph_builder.add_node("save_original", self.save_original_messages)
        graph_builder.add_node("analyze_message", self.analyze_message)
        graph_builder.add_node("process_task", self.process_task)
        graph_builder.add_node("assemble_response", self.assemble_response)

        # Define Edges
        graph_builder.set_entry_point("save_original")
        graph_builder.add_edge("save_original", "analyze_message")

        # After analysis, start processing the first task
        graph_builder.add_edge("analyze_message", "process_task")

        # Conditional edge after processing a task
        graph_builder.add_conditional_edges(
            "process_task", # Source node
            self.has_more_tasks, # Condition function
            {
                "process_task": "process_task",      # If more tasks, loop back
                "assemble_response": "assemble_response" # If no more tasks, assemble
            }
        )

        # Final edge to end the graph
        graph_builder.add_edge("assemble_response", END)

        logging.info("Graph built successfully.")
        return graph_builder.compile()


    async def run(self, content: str) -> Dict[str, Any]:
        """运行任务分析"""
        initial_state = {
            "messages": [HumanMessage(content=content)],
            # Initialize other state components
            "tasks": [],
            "current_task_index": 0,
            "task_results": {},
            "processed_tasks": set(),
            "original_messages": [] # Will be populated by the first node
        }

        logging.info(f"Starting agent run with input: {content}")
        try:
            # Stream or invoke the graph
            final_state = await self.graph.ainvoke(initial_state, {"recursion_limit": 15}) # Add recursion limit

            logging.info("Agent run completed.")
            # Return the final messages or the whole state
            return final_state # Contains the final 'messages' list

        except Exception as e:
            # Catch potential errors during graph execution (e.g., recursion limits)
            logging.error(f"任务分析执行失败: {e}", exc_info=True)
            return {
                "messages": initial_state["messages"] + [AIMessage(content=f"An error occurred during execution: {str(e)}")],
                "error": str(e)
            }

# Example Usage (if running this script directly)
# import asyncio
# logging.basicConfig(level=logging.INFO)

# async def main():
#     agent = TaskAnalyzerAgent()
#     # Example 1: Tool use
#     result1 = await agent.run("What's the weather in London and what time is it?")
#     print("\n--- Result 1 ---")
#     print(result1.get('messages', [])[-1].content) # Print last message

#     # Example 2: No tool use
#     result2 = await agent.run("Tell me a short story about a robot.")
#     print("\n--- Result 2 ---")
#     print(result2.get('messages', [])[-1].content)

#     # Example 3: Mixed
#     result3 = await agent.run("Summarize the concept of photosynthesis and tell me the current time.")
#     print("\n--- Result 3 ---")
#     print(result3.get('messages', [])[-1].content)

# if __name__ == "__main__":
#     asyncio.run(main())