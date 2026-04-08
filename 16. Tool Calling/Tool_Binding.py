from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests

#tool Create
 
@tool
def multiply(a: int, b: int) -> int:
    '''Given 2 numbers a and b, this tool returns their product'''
    return a*b



#tool binding : Process of connecting tool with llm

model = ChatGroq(model='llama-3.3-70b-versatile')
llm_with_tool = model.bind_tools([multiply])

#Not every llm has tool binding capabalities 