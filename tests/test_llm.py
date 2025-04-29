from src.llm import llm

def test_llm() -> None:
  llm_instance = llm.LLMFactory.create_llm(provider="deepseek", model_name="deepseek-chat")
  print(llm_instance)