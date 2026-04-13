# 🧪 Agno Agents Study Lab

A dedicated space for exploring the **Agno** (formerly Phidata) framework. This repository is an experimental playground used to learn the nuances of building autonomous AI agents, multi-agent orchestration, and agentic RAG.

Based on the [AI Engineer formation](https://hub.asimov.academy/formacao/engenheiro-de-agentes-de-ia/) by @AsimovAcademy 

## ⚡ Quick Start with `uv`

This project uses [uv](https://github.com/astral-sh/uv) for ultra-fast Python package and project management.

### 1. Clone the repo
```bash
git clone https://github.com/GHBAlbuquerque/agno-agents-study.git
cd agno-agents-study
```

### 2. Sync dependencies
`uv` will automatically create a virtual environment and install everything from `pyproject.toml` or `requirements.txt`:
```bash
uv sync
```

### 3. Run an experiment
Use `uv run` to execute scripts without manually activating the environment:
```bash
uv run python experiments/01_web_search.py
```

---

## 🎯 Learning Roadmap

The goal is to move progressively through Agno's core features:

1.  **Level 1: Basic Agents** - Using `Agent` with basic tools like `DuckDuckGo` or `YFinance`.
2.  **Level 2: Knowledge & RAG** - Implementing `Assistant` with a `KnowledgeBase` (PDFs, Websites) and Vector DBs.
3.  **Level 3: Memory & Session** - Exploring how to persist state using `Storage` backends.
4.  **Level 4: Multi-Agent Teams** - Orchestrating specialized agents (e.g., a "Manager" agent delegating to "Researchers").
5.  **Level 5: Agno UI** - Monitoring and debugging agent reasoning via the Agno dashboard. (#TODO)

## 🛠️ Configuration

Create a `.env` file in the root:

```env
GROQ_API_KEY=groq-***
OPENAI_API_KEY=sk-***
# If exploring different LLMs
ANTHROPIC_API_KEY=sk-ant-***
# If using specific search tools
TAVILY_API_KEY=tvly-***
```

## 🧠 Study Notes
* **Agno Philosophy:** Agents are essentially LLMs with a system prompt, a set of tools, and a memory loop.
* **Performance:** Using `uv` significantly reduces the friction of adding new libraries like `pgvector` or `pydantic` during exploration.