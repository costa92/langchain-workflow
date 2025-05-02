from llm import llm
from llm.llm import init_ollama
import pytest
from ollama._types import ResponseError

def test_llm() -> None:
  llm_instance = llm.LLMFactory.create_llm(provider="deepseek", model_name="deepseek-chat")
  print(llm_instance)
def test_init_ollama_basic():
    """测试 Ollama LLM 的基础初始化和响应"""
    try:
        llm_instance = init_ollama()
        result = llm_instance.invoke("你好，Ollama！")
        print(result)
        assert isinstance(result, str)
        assert len(result) > 0
    except ResponseError as e:
        if e.status_code == 502:
            pytest.skip("Ollama服务不可用或模型未下载 (502错误)")
        raise

def test_init_ollama_custom_model():
    """测试 Ollama LLM 的自定义模型名和配置"""
    custom_model = "deepseek-coder:6.7b"  # 使用更常见的模型名称
    custom_config = {"base_url": "http://localhost:11434/v1","api_key":"ollama"}  # 简化配置
    
    try:
        llm_instance = init_ollama(model_name=custom_model, config=custom_config)
        assert llm_instance is not None
        assert hasattr(llm_instance, "invoke")
        response = llm_instance.invoke("测试自定义模型")
        assert isinstance(response, str)
        assert len(response) > 0
    except ResponseError as e:
        if e.status_code == 502:
            pytest.skip("Ollama服务不可用或模型未下载 (502错误)")
        raise