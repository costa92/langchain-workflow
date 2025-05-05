#! /usr/bin/env python3

from agent.translation_angent import TranslationAgent

def test_translation():
    agent = TranslationAgent()
    result = agent.run("你好，世界！", "en")
    print(result["translated_text"])

