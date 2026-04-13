from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.models.openai import OpenAIChat
from agno.db.sqlite import SqliteDb

from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.chunking.recursive import RecursiveChunking

from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.vectordb.chroma import ChromaDb

import os
from dotenv import load_dotenv


load_dotenv()

# ======================
# In-memory SQL DB
# ======================

db = SqliteDb(db_file="tmp/data.db")

# ======================
# Vector DB (RAG)
# ======================

vector_db = ChromaDb(
    collection="company_reports",
    path="tmp/Chromadb",
    embedder=OpenAIEmbedder(
        id="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    ),
    persistent_client=True
)

# ======================
# Knowledge class - Manages knowledge for AI agents
# ======================

knowledge = Knowledge(
    vector_db=vector_db
)

# ======================
# Reader - Converts files, URLs, and text into searchable documents.
# ======================

pdf_reader = PDFReader(
    chunking_strategy=RecursiveChunking(
        chunk_size=2000,
        overlap=200,
    ),
)

knowledge.add_content(
    path="files/UBISOFT/",
    reader=pdf_reader,
    skip_if_exists=True
)

# ======================
# Agent
# ======================

agent = Agent(
    name="financial_analyst",
    model=OpenAIChat(
        id="gpt-5-nano",
        api_key=os.getenv("OPENAI_API_KEY")
    ),
    tools=[YFinanceTools()],
    instructions=(
        "You are a financial analysit with multiple clients",
        "Remember each client and their preferences",
    ),
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    enable_agentic_memory=True,
    add_memories_to_context=True,

    # RAG atualizado
    knowledge=knowledge,
    search_knowledge=True
)

# ======================
# Tests
# ======================

# Agent Response Script

# User 1: Analyst - Prefers concise tables
agent.print_response(
    "Hello, I prefer responses in table format, I like having concise information.", 
    session_id="ubisoft_session_1", 
    user_id="analyst_ubisoft_1"
)

# User 2: Analyst - Prefers detailed text
agent.print_response(
    "Hello, I prefer responses in text format. I like having plenty of details.", 
    session_id="ubisoft_session_2", 
    user_id="analyst_ubisoft_2"
)

agent.print_response(
    "What were Ubisoft's IRFS 15 Sales in Q1 of 2025-26 and 2025-24?",
    session_id="ubisoft_session_3", 
    user_id="analyst_ubisoft_1"
)

agent.print_response(
    "What was said of the strategic Vantage studios investments?",
    session_id="ubisoft_session_4", 
    user_id="analyst_ubisoft_2"
)
