#! /usr/bin/env python3

from langchain_core.tools import tool
from datetime import datetime


@tool
async def get_current_weather(location: str) -> str:
    """Get the current weather in a given location"""
    # print("get_current_weather")
    return f"The current weather in {location} is sunny with a temperature of 70 degrees and humidity of 50%."


@tool
async def get_current_time() -> str:
    """Get the current time"""
    # print("get_current_time")
    return f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# --- Tool Definition ---
@tool
def calculator(expression: str) -> str:
    """
    Calculates the result of a mathematical expression.
    Use this tool for any numerical calculations like addition, subtraction, multiplication, division, percentages, etc.
    Input should be a valid mathematical expression string (e.g., '100 * 1.1', '50 / 2', '(100 - 50) / 50 * 100').
    """
    print(f"--- Calling Calculator Tool with expression: {expression} ---")
    try:
        # WARNING: eval is potentially unsafe with untrusted input.
        # Consider using a safer alternative like numexpr or ast.literal_eval for production.
        # For simplicity in this example, we use eval.
        result = eval(expression, {"__builtins__": None}, {}) # Basic safety
        return f"The result of '{expression}' is {result}"
    except Exception as e:
        print(f"--- Calculator Error: {e} ---")
        return f"Error calculating '{expression}': {str(e)}. Please provide a valid mathematical expression."
