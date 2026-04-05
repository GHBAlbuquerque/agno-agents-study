from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.yfinance import YFinanceTools
from agno.os import AgentOS
from agno.db.sqlite import SqliteDb

from dotenv import load_dotenv
load_dotenv()

# Setup DB
db = SqliteDb(db_file="tmp/data.db")

agent = Agent(
    name="financial_analyst",
    model=OpenAIResponses(id="gpt-5.4-nano", 
                          reasoning_effort="low",
                          parallel_tool_calls=False),
    tools=[YFinanceTools()],
    instructions=["Search for the stock price using the provided tools.",
        "Check the stock exchange for each ticker and identify the correct local currency (e.g., EUR for Euronext, JPY for Tokyo, USD for NASDAQ).",
        "Display the final stock information in a markdown table, ensuring you include the correct currency symbol and currency code next to the price.",
        "Be concise, but ensure the tool call finishes successfully."],
    db=db,
    add_history_to_context=True,
    num_history_runs=3
)

agent_os = AgentOS(
    name="financial-agent-os",
    description="My financial Agent OS",
    agents=[agent]
    )

app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve("1_3_agentos_v2:app", reload=True)