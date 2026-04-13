from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.models.openai import OpenAIChat
from agno.db.sqlite import SqliteDb

from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.chunking.recursive import RecursiveChunking

from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.vectordb.chroma import ChromaDb

from agno.team.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools

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
# Agents
# ======================

news_agent = Agent(
    name="news_agent",
    model=OpenAIChat(id="gpt-5-nano"),
    role="You are a news researcher.",
    instructions=[
        "Use your search tools to find information on the web about companies listed on the NYSE and NASDAQ",
    ],
    tools=[DuckDuckGoTools(enable_search=False, enable_news=True)],
    markdown=True
)

financial_agent = Agent(
    name="financial_analyst",
    model=OpenAIChat(
        id="gpt-5-nano"
    ),
    tools=[YFinanceTools()],
    instructions=("You are a financial analyst."),
    markdown=True,
)

reports_agent = Agent(
    name="financial_analyst",
    model=OpenAIChat(
        id="gpt-5-nano"
    ),
    instructions=( "You are a financial report analyst."),
    
    # RAG
    knowledge=knowledge,
    search_knowledge=True,
    add_knowledge_to_context=True,
    markdown=True,
)

# ======================
# Team
# ======================

multi_agent_team = Team(
    name="Team Analyst",
    model=OpenAIChat(
        id="gpt-5-nano"
    ),
    members=[news_agent, financial_agent, reports_agent],
    instructions=[
        "You must understand the information requested by the user and provide an appropriate response."
        "To obtain information about balance sheets and income statements (P&L), use the reports_agent."
        "To obtain information about quotes/stock prices, use the financial_agent."
        "To obtain information about news, use the news_agent."
    ],
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    show_members_responses=True,
    get_member_information_tool=True,
    add_datetime_to_context=True,
    markdown=True,
)

# ======================
# Test
# ======================

multi_agent_team.print_response("What is the Ubisoft stock price, and what news might have moved the price in the last few days?", session_id="ubisoft_session_5", user_id="ubisoft_analyst")