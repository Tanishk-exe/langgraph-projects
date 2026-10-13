from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
import requests
import os

load_dotenv()

os.environ['LANGCHAIN_PROJECT']='HITL'

@tool
def get_stock(symbol: str) -> dict:
    """returns the latest price of the stock"""
    url=f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=LDJQLEJGNWNWUJS2'
    r=requests.get(url)
    return r.json()

@tool
def buy_stocks(symbol: str, quantity:int, total_price:float) -> str:
    """Buy the given stocks and according to its price"""
    decision=interrupt(f'Approve buying {quantity} {symbol} stocks for this ${total_price:.2f}?')

    if decision == 'yes':
        return f'You Bought {quantity} {symbol} stocks at the total proce of ${total_price:.2f}.'
    else:
        return f'Buying Declined'

model=ChatOpenAI(model='gpt-4o-mini')


tools=[get_stock, buy_stocks]

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

check=InMemorySaver()
wf=graph.compile(checkpointer=check)

config={'run_name':'HITL', 'configurable':{'thread_id':'thread2'}}
print("\n")
print("----------------------------- HITL Stock Agent -----------------------------\n")
inp=input("You: ")
rs=wf.invoke({'messages':[HumanMessage(content=inp)]}, config=config)
print("AI: ",rs['messages'][-1].content)
inp2=input("You: ")
rs1=wf.invoke({'messages':[HumanMessage(content=inp2)]}, config=config)
print("HITL: ",rs1.get("__interrupt__")[0].value)

decision=input("Approve(Yes/No): ")
rs2=wf.invoke(Command(resume=decision), config=config)
print(rs2['messages'][-1].content,"\n")
