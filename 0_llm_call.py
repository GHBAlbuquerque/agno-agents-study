from agno.models.groq import Groq
from agno.models.message import Message

from dotenv import load_dotenv
load_dotenv()

model=Groq(id="llama-3.3-70b-versatile")

# User msg
user_msg = Message(
    role="user",
    content=[{"type":"text", "text": "Hello, my name is Gigi"}],
)

# Assistant msg
assistant_msg = Message(
    role="assistant",
    content=[{}]
)


# Invoke 
response = model.invoke(
    messages=[user_msg],
    assistant_message=assistant_msg
    )