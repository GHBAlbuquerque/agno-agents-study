from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.models.groq import Groq

from dotenv import load_dotenv
load_dotenv()


agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools()],
    instructions=["Search for the stock price using the provided tools.",
        "Display the final stock information in a markdown table.",
        "Be concise, but ensure the tool call finishes successfully."],
    #debug_mode=True
)

agent.print_response("What is Square Enix current stock price in dollars?", stream=True)