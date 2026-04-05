from agno.agent import Agent
from agno.tools.tavily import TavilyTools
from agno.models.openai import OpenAIChat

from dotenv import load_dotenv
load_dotenv()

def celsius_to_fh(temperature_celsius: float):
    """"
    Converts temperature from Celsius to Fahrenheit.

    Args:
        temperature_celsius (float): Temperature in degrees Celsius

    Returns:
        float: Temperature in Fahrenheit
    """
    return (temperature_celsius * 9/5) + 32



agent = Agent(
    model=OpenAIChat(id="gpt-5.4-nano"),
    tools=[TavilyTools(),
           celsius_to_fh],
    debug_mode=True,
)

agent.print_response("Use your tools to search the temperature today in São Paulo city in Fahrenheit.")