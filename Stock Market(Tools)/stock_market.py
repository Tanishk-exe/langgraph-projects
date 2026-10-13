from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
import requests
import os

load_dotenv()

@tool
def get_stock(symbol: str) -> dict:
    """returns the latest price of the stock"""
    url=f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=LDJQLEJGNWNWUJS2'
    r=requests.get(url)
    return r.json()

model=ChatOpenAI(model='gpt-5-mini')

duck=DuckDuckGoSearchRun()
tools=[get_stock, duck]
llm_binded=model.bind_tools(tools)

class mssg(TypedDict):
    messages:Annotated[list[BaseMessage], add_messages]

def chatnode(state:mssg):
    rs=llm_binded.invoke(state['messages'])
    return {'messages':[rs]}

toolnode=ToolNode(tools)

graph=StateGraph(mssg)

graph.add_node('chatnode', chatnode)
graph.add_node('tools', toolnode)

graph.add_edge(START, 'chatnode')

graph.add_conditional_edges('chatnode', tools_condition)

graph.add_edge('tools', 'chatnode')

wf=graph.compile()

config={'run_name':'tools chat', 'tags':['duckducksearch', 'stock tool']}
print("\nStock AI - Ask the current price rates of Stock!!!")
inp=input("You: ")
rs=wf.invoke({'messages':[HumanMessage(content=inp)]}, config=config)
print("Stock AI: ",rs['messages'][-1].content)
