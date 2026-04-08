'''It is process where llm decides, that it needs to call a tool(function), llm will not
   run the tool by itself, it just suggest to use that, we need to call the tool by our 
   side using langchain.'''

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

#tool Create
 
@tool
def multiply(a: int, b: int) -> int:
    '''Given 2 numbers a and b, this tool returns their product'''
    return a*b



#tool binding : Process of connecting tool with llm

model = ChatGroq(model='llama-3.3-70b-versatile')
llm_with_tool = model.bind_tools([multiply])



#tool calling : process where llm decides, that it needs to call a tool(function)

print(llm_with_tool.invoke('can you multipy 7 with 9'))

