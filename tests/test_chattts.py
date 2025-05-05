import torch
import ChatTTS
from IPython.display import Audio
import scipy
import requests

chat = ChatTTS.Chat()
# chat.load_models() # Uncommented to load models before inference
chat.download_models()

from llm.chat_tts_llm import llm_api
from llm.chat_tts_llm import OllamaLlama3API

user_question = '用中英文混杂的方式，简单介绍宁波'

# Initialize Ollama API client with error handling
ollama_api = OllamaLlama3API(base_url="http://localhost:11434", model="llama3")

try:
    text = ollama_api.call(user_question, prompt_version='deepseek')
    text = ollama_api.call(text, prompt_version='deepseek_TN')
    print(text)
except requests.exceptions.JSONDecodeError as e:
    print(f"Error decoding JSON response from Ollama API: {e}")
    print("Please check if Ollama server is running at http://localhost:11434")
    raise
except Exception as e:
    print(f"Unexpected error calling Ollama API: {e}")
    raise

# Load speaker stats (ensure path is correct)
try:
    # Adjust path if your assets are not in a direct 'asset' subdir
    spk_stat_path = 'asset/spk_stat.pt'
    spk_stat = torch.load(spk_stat_path)
except FileNotFoundError:
    print(f"Error: Speaker stats file not found at {spk_stat_path}")
    # Handle error - maybe exit or use a default?
    raise SystemExit(f"Missing required file: {spk_stat_path}")

rand_spk = torch.randn(768) * spk_stat.chunk(2)[0] + spk_stat.chunk(2)[1]

params_infer_code = {'spk_emb': rand_spk, 'temperature': .3}
params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}
# 将infer函数中的文本参数替换为从LLM获取的text
# Check if LLM text generation was successful before inferring
if 'text' in locals() and text:
    wav = chat.infer(text,
                     params_refine_text=params_refine_text,
                     params_infer_code=params_infer_code)

    # IPython display might not work in a standard .py script
    # from IPython.display import Audio
    # Audio(wav[0], rate=24_000, autoplay=True)

    # Save the audio
    output_filename = "./chattts_download.wav"
    scipy.io.wavfile.write(filename=output_filename, rate=24_000, data=wav[0].T)
    print(f"Audio saved to {output_filename}")
else:
    print("Error: Model not loaded properly. Please ensure the model is initialized correctly before inference.")
    raise AssertionError("Model initialization failed - check if model weights are loaded properly")