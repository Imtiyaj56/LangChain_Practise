""" 
    Force LLM To Send Output In Json Format Even If LLM Dont Support It.
    Ex : Hugging Face LLMs(Mostly)
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Meta-Llama-3-8B-Instruct',
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template= 'Give me 5 facts about {topic}. \n {format_instruction}',
    input_variables= ['topic'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'topic': 'Indian Independence'})
print(result)
print(type(result))

"""
    Disadvantage : It does not able to enforce schema of json pf our choice, Json format(schema)
                   is only decided by LLLM.
    Solution : This Problem is solved by "StructuredOutputParser".
"""