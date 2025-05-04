#!/usr/bin/env python3

from gtts import gTTS

def test_gTTS():
  text = "这是一段普通的文本，我是一个Python包。"
  tts = gTTS(text=text, lang='zh-cn', slow=False)
  # 将文本转为语音并保存为音频文件
  tts.save("output.mp3")
