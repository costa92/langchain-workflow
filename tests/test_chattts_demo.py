

import ChatTTS
import torch
import torchaudio

chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance

texts = ["用中英文混杂的方式，简单介绍宁波"]

wavs = chat.infer(texts)

print(wavs)
# # 转换 numpy → tensor，并添加 batch 维度 [1, samples]
waveform = torch.from_numpy(wavs[0]).unsqueeze(0).float()
# print(waveform)
# # 保存为 WAV（明确指定 format）
torchaudio.save("output1.wav", waveform, sample_rate=24000, format="wav")

# wavs = chat.infer(texts)

# torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)