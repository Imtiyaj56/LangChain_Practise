'''When Langchain suggest us to use specific tool, we need to execute that tool
   from our side only, This process is called Tool Execution.'''

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

query = HumanMessage('can you multipy 7 with 9')     
messages = [query]    #Human Message


result = llm_with_tool.invoke(messages)
messages.append(result)   #AI Message


tool_result = multiply.invoke(result.tool_calls[0])
messages.append(tool_result)   #Tool Message

print(llm_with_tool.invoke(messages).content)

