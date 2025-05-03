from llm import llm
from llm.llm import init_ollama
import pytest
from ollama._types import ResponseError
from langchain.schema import HumanMessage
def test_llm() -> None:
  llm_instance = llm.LLMFactory.create_llm(provider="deepseek", model_name="deepseek-chat")
  print(llm_instance)
def test_init_ollama_basic():
    """测试 Ollama LLM 的基础初始化和响应"""
    try:
        llm_instance = init_ollama()
        message = HumanMessage(content="你好，Ollama！")
        result = llm_instance.invoke([message])
        print(result)
    except ResponseError as e:
        if e.status_code == 502:
            pytest.skip("Ollama服务不可用或模型未下载 (502错误)")
        raise

def test_init_ollama_custom_model():
    """测试 Ollama LLM 的自定义模型名和配置"""
    custom_model = "deepseek-coder:6.7b"  # 使用更常见的模型名称
    custom_config = {"base_url": "http://localhost:11434","api_key":"ollama"}  # 简化配置
    
    try:
        llm_instance = init_ollama(model_name=custom_model, config=custom_config)
        assert llm_instance is not None
        assert hasattr(llm_instance, "invoke")
        response = llm_instance.invoke("func main(){fmt.Println(\"Hello, World!\")}")
        print(response)
    except ResponseError as e:
        if e.status_code == 502:
            pytest.skip("Ollama服务不可用或模型未下载 (502错误)")
        raise