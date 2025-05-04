import asyncio
import edge_tts
import os
import time
TEXT = "Hello, world!"
# 可以通过 `edge-tts --list-voices` 命令查看所有可用语音
# 选择一个中文语音, 例如 "zh-CN-XiaoxiaoNeural" 或 "zh-CN-YunxiNeural"
# VOICE = "zh-CN-XiaoxiaoNeural"
VOICE = "en-US-JennyNeural"   
OUTPUT_FILE = "output_edge.mp3"


async def test_edge_tts():
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_FILE)
    print(f"语音文件 '{OUTPUT_FILE}' 已成功保存！")
    time.sleep(10)

