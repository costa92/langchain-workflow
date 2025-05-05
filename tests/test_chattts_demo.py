

import torch
import ChatTTS
from IPython.display import Audio

# chat = ChatTTS.Chat()
chat = ChatTTS.Chat()

chat.download_models(source='local')

text = "用中英文混杂的方式，简单介绍宁波"
wav = chat.infer(text)
Audio(wav[0], rate=24_000, autoplay=True)