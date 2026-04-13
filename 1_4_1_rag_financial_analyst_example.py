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
# Banco de memória
# ======================

db = SqliteDb(db_file="tmp/data.db")


# ======================
# Vector DB (RAG)
# ======================

vector_db = ChromaDb(
    collection="empresas_relatorios",
    path="tmp/Chromadb",
    embedder=OpenAIEmbedder(
        id="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    persistent_client=True,
)

knowledge = Knowledge(
    vector_db=vector_db,
)

# ======================
# Reader (ATUALIZADO)
# ======================

pdf_reader = PDFReader(
    chunking_strategy=RecursiveChunking(
        chunk_size=2000,
        overlap=200,
    ),
)

knowledge.add_content(
    path="files/PETR/",
    reader=pdf_reader,
    skip_if_exists=True
)

knowledge.add_content(
    path="files/VALE/",
    reader=pdf_reader,
    skip_if_exists=True
)


# ======================
# Agent
# ======================

agent = Agent(
    name="analista_financeiro",
    model=OpenAIChat(
        id="gpt-5-nano",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    tools=[YFinanceTools()],
    instructions=(
        "Voce e um analista financeiro e tem diferentes clientes. "
        "Lembre-se de cada cliente e suas preferencias."
    ),
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    enable_user_memories=True,
    add_memories_to_context=True,
    enable_agentic_memory=True,

    # RAG atualizado
    knowledge=knowledge,
    search_knowledge=True,  # padrão mais atual
)


# ======================
# Testes
# ======================

agent.print_response(
    "Olá, qual foi o lucro liquido da Petrobras no 2T25?"
)

agent.print_response(
    "O que foi comentado sobre o CAPEX da Vale no 2T25?"
)
