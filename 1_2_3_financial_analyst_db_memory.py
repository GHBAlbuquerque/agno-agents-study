import sys
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.yfinance import YFinanceTools
from agno.db.sqlite import SqliteDb

from dotenv import load_dotenv
load_dotenv()

# --- ERROR SUPPRESSION BLOCK ---
def mute_httpx_garbage_collection_error(unraisable):
    if unraisable.exc_type == AttributeError and "'SyncHttpxClientWrapper' object has no attribute '_state'" in str(unraisable.exc_value):
        return  
    sys.__unraisablehook__(unraisable) 

# Override the default hook
sys.unraisablehook = mute_httpx_garbage_collection_error
# -------------------------------

# Setup DB
db = SqliteDb(db_file="tmp/data.db")

agent = Agent(
    name="financial_analyst",
    model=OpenAIResponses(id="gpt-5.4-nano", 
                          reasoning_effort="low",
                          parallel_tool_calls=False),
    tools=[YFinanceTools()],
    instructions=["You are a financial analyst with different clients."
                  "Remember each client, their informations and preferences."],
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    enable_user_memories=True,
    add_memories_to_context=True,
    enable_agentic_memory=True
)

# Agent Response Script

# User 1: Electronic Arts Analyst - Prefers concise tables
agent.print_response(
    "Hello, I prefer responses in table format, I like having concise information.", 
    session_id="electronic_arts_session_1", 
    user_id="analyst_electronic_arts"
)

# User 2: Square Enix Analyst - Prefers detailed text
agent.print_response(
    "Hello, I prefer responses in text format. I like having plenty of details.", 
    session_id="square_enix_session_1", 
    user_id="analyst_square_enix"
)

# Example Queries
agent.print_response("What is the stock price of Electronic Arts?", session_id="electronic_arts_session_2", user_id="analyst_electronic_arts")
agent.print_response("What is the stock price of Square Enix (9684.T)?", session_id="square_enix_session_2", user_id="analyst_square_enix")
agent.print_response("What was the total revenue and net income for Square Enix in 2020 (Fiscal year 2020)?", session_id="square_enix_session_2", user_id="analyst_square_enix")
  