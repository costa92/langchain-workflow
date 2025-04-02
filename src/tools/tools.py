#! /usr/bin/env python3

from langchain_core.tools import tool
from datetime import datetime


@tool
def get_current_weather(location: str) -> str:
    """Get the current weather in a given location"""
    print("get_current_weather")
    return f"The current weather in {location} is sunny with a temperature of 70 degrees and humidity of 50%."


@tool
def get_current_time() -> str:
    """Get the current time"""
    print("get_current_time")
    return f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
