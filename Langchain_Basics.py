from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")

model = init_chat_model("gemini-2.5-flash",model_provider="google-genai")

messages = [
    SystemMessage("translate this to spanish"),
    HumanMessage("Hi, how are you?")
]

ans = model.invoke(messages)
print(ans.content)

# for token in model.stream(messages):
#     print(token.content, end="|")