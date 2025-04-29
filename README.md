# python-layout-demo

## 项目概述

本项目实现了一个基于图结构的应用系统，通过异步处理技术提高了性能。项目采用模块化设计，主要由图结构核心组件和操作流程组成。

## 技术栈

- Python 3.x
- LangChain Core
- LangChain OpenAI
- 异步编程 (asyncio)
- 图结构处理
- 类型提示 (typing-extensions)

## 项目特点

- 基于图结构的任务处理系统
- 异步并行处理提升性能
- 模块化架构设计
- 强类型支持
- 完整的测试覆盖

## 项目结构

```
.
├── src/               # 源代码目录
├── tests/             # 测试用例目录
├── main.py           # 主程序入口
├── requirements.txt   # 项目依赖
├── pyproject.toml    # 项目配置
└── .env.example      # 环境变量示例
```

## 安装说明

1. 克隆项目
```bash
git clone [repository-url]
cd langgraph-workflow
```

2. 创建虚拟环境（推荐）
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

使用  conda
```
 conda create --name langgraph-workflow python=3.13
 conda activate langgraph-workflow
```

3. 安装依赖
```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

4. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，填入必要的配置信息
```

## 使用说明

1. 运行主程序

```bash
pip install -e .
python main.py
```

2. 运行测试

```bash
pytest
```

## 核心功能

- 图结构任务处理：使用图结构组织和管理任务流程
- 并行工作流：支持多任务并行处理
- 异步操作：通过异步技术提高系统响应性能
- 消息处理：支持人机对话消息的处理和响应

## 开发指南

- 遵循 PEP 8 编码规范
- 使用 mypy 进行类型检查
- 保持测试覆盖率
- 使用 git 进行版本控制
- [打包](https://packaging.pythonlang.cn/en/latest/guides/distributing-packages-using-setuptools/)

## 许可证

[添加许可证信息]

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 联系方式

[添加联系方式信息]

## FastAPI 聊天接口

### 启动 API 服务

```bash
uvicorn src.api:app --reload
```

### 聊天接口说明

- 路径: `/chat`
- 方法: `POST`
- 请求体:

```json
{
  "messages": [
    {"role": "human", "content": "你好，今天天气怎么样？"}
  ]
}
```

- 响应体:

```json
{
  "reply": "AI 回复内容",
  "messages": [
    {"role": "HumanMessage", "content": "你好，今天天气怎么样？"},
    {"role": "AIMessage", "content": "今天天气晴朗，气温适中。"}
  ]
}
```

### 说明
- `messages` 为历史消息列表，最后一条为用户输入。
- `reply` 为本次 AI 回复内容。
- 支持多轮对话。
