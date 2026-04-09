from langchain.tools import tool
import requests
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import json

load_dotenv()


#1st tool create : which fetches the factor of currency rate

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    '''
    This function fetches the currency conversion factor between a given base currency and 
    a target currency.
    '''
    url = f'https://v6.exchangerate-api.com/v6/a4a9866534b6f78014f4c596/pair/{base_currency}/{target_currency}'

    response = requests.get(url)

    # returning only conversion_rate (not full JSON)
    return response.json()['conversion_rate']


# 2nd tool create : Which converts the currency base on conversion rate....
@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    '''
    Given a currency convesrion rate, this function calculates the target currency value from
    the given base currency value
    '''

    return base_currency_value * conversion_rate


# Tool Binding

model = ChatGroq(model='llama-3.3-70b-versatile')
llm_with_tool = model.bind_tools([get_conversion_factor, convert])

#Human Message
messages = [HumanMessage("What is the conversion factor between USD and INR, and based on that convert 56 USD into INR")]  

#AI Message
ai_message = llm_with_tool.invoke(messages)
messages.append(ai_message)



conversion_rate = None

# execute first tool
for tool_call in ai_message.tool_calls:

    if tool_call['name'] == "get_conversion_factor":
        tool_message_1 = get_conversion_factor.invoke(tool_call['args'])

        
        conversion_rate = tool_message_1

        
        messages.append(
            ToolMessage(
                content=str(tool_message_1),
                tool_call_id=tool_call['id']
            )
        )


# execute second tool
for tool_call in ai_message.tool_calls:

    if tool_call['name'] == "convert":

        tool_call['args']['conversion_rate'] = conversion_rate
        tool_message_2 = convert.invoke(tool_call['args'])

        messages.append(
            ToolMessage(
                content=str(tool_message_2),
                tool_call_id=tool_call['id']
            )
        )


# final LLM response after tool execution
result = llm_with_tool.invoke(messages)

print(result.content)