from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain.chat_models import init_chat_model
import os
import ollama
from dotenv import load_dotenv
from typing import Annotated, Literal

load_dotenv()

def call_ollama(message: str, model_name: str = "llama3", system_prompt: str = "") -> str:
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": message})

    response = ollama.chat(
        model=model_name,
        messages=messages
    )

    return response['message']['content']

def call_gemini(message:str,model_name:str="gemini-2.5-flash",system_prompt:str=""):
    
    humanMessage = HumanMessage(message)
    systemMessage = SystemMessage(system_prompt)

    api_key = os.environ.get("GOOGLE_API_KEY");
    
    model = init_chat_model(model_name,model_provider="google_genai")
    messages = [
        systemMessage,humanMessage
    ]

    response = model.invoke(messages)

    return response.content

def simple_call_gemini(message:str,model_name:str="gemini-2.5-flash",system_prompt:str=""):
    

    api_key = os.environ.get("GOOGLE_API_KEY");
    
    model = init_chat_model(model_name,model_provider="google_genai")
    

    response = model.invoke(message)

    return response.content

# ans = call_ollama("Hey, who are you?")
# print(ans)

# ans=call_gemini("hey who are you?")
# print(ans)

## LANGGRAPH

from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

class State(TypedDict):
    messages:Annotated[list,add_messages]

graph_builder = StateGraph(State)

## Node
def chatbot(state:State):
    return {"messages":[simple_call_gemini(state["messages"])]}

## Adding node with name,function (that is the node)
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)

## running the graph
graph = graph_builder.compile()

user_input = input("Enter message")

## you start with some state (initial state that matches class State)
state = graph.invoke({"messages":[{"role":"user","content":user_input}]})

print(state["messages"][-1].content)

