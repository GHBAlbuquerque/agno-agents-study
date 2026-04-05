from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.os import AgentOS

from dotenv import load_dotenv
load_dotenv()

test_agent = Agent(
    name="my-test-agent",
    model=OpenAIResponses(id="gpt-5.4-nano", reasoning_effort="low"),
    instructions=["You are a helpful AI Assistant"],
)

agent_os = AgentOS(
    name="my-first-os",
    description="My first Agent OS",
    agents=[test_agent]
    )

app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve("1_3_agentos_v1:app", reload=True)