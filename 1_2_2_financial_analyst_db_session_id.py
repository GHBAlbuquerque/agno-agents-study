from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.yfinance import YFinanceTools
from agno.db.sqlite import SqliteDb

from dotenv import load_dotenv
load_dotenv()

import requests

# Currency conversion tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Converts a given amount from one currency to another using current exchange rates.
    
    Args:
        amount: The amount of money to convert.
        from_currency: The 3-letter currency code to convert from (e.g., 'USD', 'EUR').
        to_currency: The 3-letter currency code to convert to (e.g., 'BRL', 'JPY').
    """
    try:
        # Using the free Frankfurter API
        url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_currency}&to={to_currency}"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        converted_amount = data['rates'][to_currency]
        
        return f"{amount} {from_currency} is equal to {converted_amount} {to_currency}."
    except Exception as e:
        return f"Error converting currency: {e}"
    
# Setup DB
db = SqliteDb(db_file="tmp/data.db")

agent = Agent(
    # session_id="square_enix_session", -> could set here
    # user_id="user_se", -> also could be set here
    name="financial_analyst",
    model=OpenAIResponses(id="gpt-5.4-nano", 
                          reasoning_effort="low",
                          parallel_tool_calls=False),
    tools=[YFinanceTools(), convert_currency],
    instructions=["Search for the stock price using the provided tools.",
        "Check the stock exchange for each ticker and identify the correct local currency (e.g., EUR for Euronext, JPY for Tokyo, USD for NASDAQ).",
        "Display the final stock information in a markdown table, ensuring you include the correct currency symbol and currency code next to the price.",
        "Be concise, but ensure the tool call finishes successfully."],
    db=db,
    add_history_to_context=True,
    num_history_runs=3
)

agent.print_response("What is Square Enix current stock price in dollars?", stream=True, session_id="square_enix_session", user_id="user_se")
agent.print_response("What is Ubisoft current stock price in dollars?", stream=True, session_id="ubisoft_session", user_id="user_ubisoft")
agent.print_response("What is Electronic Arts stock price in dollars?", stream=True, session_id="ea_session", user_id="user_ea")
agent.print_response("How many companies stock prices have we searched for?", stream=True, session_id="square_enix_session", user_id="user_se") #will only remember what was searched in SE session