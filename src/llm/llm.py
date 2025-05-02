#! /usr/bin/env python3

# llm_factory.py

import os
from typing import Optional, Dict, Any, Callable, Type
# 移除 functools.lru_cache，因为缓存客户端实例可能会导致问题
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from dotenv import load_dotenv


# 加载 .env 文件环境变量
load_dotenv()

class LLMFactory:
    """
    统一的 LLM 工厂类，支持 Azure、OpenAI、DeepSeek、Anthropic
    以及自定义注册的 LLM 提供商。
    """

    # 自定义 LLM 提供商注册表
    _custom_providers: Dict[str, Callable[[str, Dict[str, Any], Dict[str, Any]], BaseChatModel]] = {}

    @classmethod
    def register_provider(cls, provider_name: str, provider_func: Callable[[str, Dict[str, Any], Dict[str, Any]], BaseChatModel]) -> None:
        """
        注册自定义 LLM 提供商。

        参数:
            provider_name: 提供商名称（不区分大小写）。
            provider_func: 创建 LLM 实例的函数。
                           接收 model_name、config 和 kwargs 参数。
        """
        if not callable(provider_func):
            raise TypeError("provider_func 必须是可调用的")
        cls._custom_providers[provider_name.lower()] = provider_func

    @classmethod
    def create_llm(
        cls,
        provider: str,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> BaseChatModel:
        """
        创建指定提供商的 LLM 实例。

        配置优先级: kwargs > config 字典 > 环境变量 > 默认值。

        参数:
            provider: LLM 提供商名称（如 "azure", "openai", "custom_provider"）。不区分大小写。
            model_name: 提供商的特定模型名称或部署名称。
            config: 可选字典，包含 api_key、base_url 等配置。
            **kwargs: 直接传递给底层 LangChain 模型构造函数的额外关键字参数，会覆盖其他设置。

        返回:
            BaseChatModel 子类的实例。

        异常:
            ValueError: 如果提供商不受支持或未注册。
            TypeError: 如果注册的自定义提供商函数不可调用（在注册期间）。
        """
        provider_lower = provider.lower()
        config = config or {}
        
        # 默认温度值
        default_temperature = 0.0

        # 1. 首先检查已注册的自定义提供商
        if provider_lower in cls._custom_providers:
            custom_func = cls._custom_providers[provider_lower]
            # 确保它仍然是可调用的，以防注册表被不当修改
            if not callable(custom_func):
                 raise TypeError(f"'{provider}'的注册提供商函数不可调用。")
            return custom_func(model_name, config, kwargs)

        # 2. 处理内置提供商
        provider_params: Dict[str, Any] = {}
        ModelClass: Optional[Type[BaseChatModel]] = None

        # 根据提供商类型设置参数
        if provider_lower == "azure":
            ModelClass = AzureChatOpenAI
            provider_params = {
                "azure_deployment": model_name,
                "openai_api_version": config.get("api_version", os.getenv("OPENAI_API_VERSION")),
                "azure_endpoint": config.get("base_url", os.getenv("AZURE_OPENAI_ENDPOINT")),
                "api_key": config.get("api_key", os.getenv("AZURE_OPENAI_API_KEY")),
                "temperature": config.get("temperature", default_temperature),
            }
        elif provider_lower == "openai":
            ModelClass = ChatOpenAI
            provider_params = {
                "model": model_name,
                "base_url": config.get("base_url", os.getenv("OPENAI_API_BASE")),
                "api_key": config.get("api_key", os.getenv("OPENAI_API_KEY")),
                "temperature": config.get("temperature", default_temperature),
            }
        elif provider_lower == "deepseek":
            ModelClass = ChatDeepSeek
            provider_params = {
                "model": model_name,
                "base_url": config.get("base_url", os.getenv("DEEPSEEK_API_BASE")),
                "api_key": config.get("api_key", os.getenv("DEEPSEEK_API_KEY")),
                "temperature": config.get("temperature", default_temperature),
            }
        elif provider_lower == "anthropic":
            ModelClass = ChatAnthropic
            provider_params = {
                "model": model_name,
                "base_url": config.get("base_url", os.getenv("ANTHROPIC_API_BASE")),
                "api_key": config.get("api_key", os.getenv("ANTHROPIC_API_KEY")),
                "temperature": config.get("temperature", default_temperature),
            }
        elif provider_lower == "volcengine":
            ModelClass = ChatOpenAI
            provider_params = {
                "model": model_name,
                "base_url": config.get("base_url", os.getenv("VOLCENGINE_API_BASE")),
                "api_key": config.get("api_key", os.getenv("VOLCENGINE_API_KEY")),
                "temperature": config.get("temperature", default_temperature),
            }
        elif provider_lower == "ollama":
            ModelClass = ChatOllama
            provider_params = {
                "model": model_name,
                "base_url": config.get("base_url", os.getenv("OLLAMA_API_BASE")),
                "api_key": config.get("api_key", os.getenv("OLLAMA_API_KEY")),
            }
        else:
            supported = ["azure", "openai", "deepseek", "anthropic", "volcengine"] + list(cls._custom_providers.keys())
            raise ValueError(f"不支持的提供商: '{provider}'。支持的提供商有: {supported}")

        # 合并参数：从提供商特定参数开始，然后用 kwargs 覆盖
        final_params = {**provider_params, **kwargs}

        # 实例化模型
        # 确保 ModelClass 已分配
        if ModelClass is None:
             # 由于上面的 ValueError，理论上不应该到达这里，
             # 但提供额外的保障并满足类型检查器。
             raise RuntimeError(f"提供商 '{provider_lower}' 的 ModelClass 未设置。这表明内部逻辑错误。")

        return ModelClass(**final_params)

# 初始化 LLM 实例
def init_llm(provider: str, model_name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> BaseChatModel:
    """
    初始化任意提供商的 LLM 实例
    
    参数:
        provider: 提供商名称
        model_name: 模型名称
        config: 配置字典
        kwargs: 额外参数
        
    返回:
        LLM 实例
    """
    return LLMFactory.create_llm(provider, model_name, config, **kwargs)

# 初始化 DeepSeek LLM 实例
def init_deepseek(model_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None, **kwargs) -> BaseChatModel:
    """
    初始化 DeepSeek LLM 实例
    
    参数:
        model_name: 模型名称，默认为 "deepseek-chat"
        config: 配置字典
        kwargs: 额外参数
        
    返回:
        DeepSeek LLM 实例
    """
    model_name = model_name or "deepseek-chat"
    return LLMFactory.create_llm("deepseek", model_name, config, **kwargs)

# 初始化 volcengine_chat
def get_volcengine_chat_llm(model_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None, **kwargs) -> BaseChatModel:
    """
    初始化火山引擎 LLM 实例
    
    参数:
        model_name: 模型名称，默认为 "deepseek-v3-250324"
        config: 配置字典
        kwargs: 额外参数
        
    返回:
        火山引擎 LLM 实例
    """
    model_name = model_name or "deepseek-v3-250324"
    return LLMFactory.create_llm("volcengine", model_name, config, **kwargs)

# 初始化 ollama
def init_ollama(model_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None, **kwargs) -> BaseChatModel:
    """
    初始化 ollama 实例
    """
    model_name = model_name or "llama3.1:8b"
    return LLMFactory.create_llm("ollama", model_name, config, **kwargs)