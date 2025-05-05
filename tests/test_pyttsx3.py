#!/usr/bin/env python3
import pyttsx3

def test_pyttsx3():
    # 初始化 TTS 引擎
  engine = pyttsx3.init()

  # (可选) 查看和设置属性
  voices = engine.getProperty('voices')
  # 打印可用的语音
  # for voice in voices:
  #     print(f"ID: {voice.id}, Name: {voice.name}, Lang: {voice.languages}")

  # 尝试查找并设置中文语音 (ID 可能因系统而异)
  # 在 Windows 上, 中文语音通常包含 "Chinese" 或 "Huihui" 等字样
  # 你需要根据上面打印出的列表选择合适的 voice.id
  # engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ZH-CN_HUIHUI_11.0') # Windows 示例

  rate = engine.getProperty('rate')   # 获取当前语速
  engine.setProperty('rate', rate - 50) # 减慢语速

  volume = engine.getProperty('volume') # 获取当前音量 (0.0 to 1.0)
  engine.setProperty('volume', 0.9)   # 设置音量

  # 要转换的文本
  text = "龙汐瑶，还看手机，还不洗澡睡觉"

  # 让引擎朗读文本
  engine.say(text)

  # 等待语音播放完成
  engine.runAndWait() 

  engine.stop() # 停止引擎 (可选，但在脚本结束时是好习惯)

  print("语音已播放完毕。")

  # (可选) 保存为文件 (注意：并非所有引擎都完美支持保存为文件)
  # engine.save_to_file(text, 'output_pyttsx3.mp3')
  # engine.runAndWait()
  # print("语音文件已尝试保存。")