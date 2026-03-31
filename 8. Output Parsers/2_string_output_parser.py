''' in first program we have not used string_output_parser
but in this code we will do same thing but using string_output_parser,
and then compare the output of two, and see the need of string_output_parser.'''

from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Meta-Llama-3-8B-Instruct',
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

# 1st Prompt --> Detailed Prompt
template1 = PromptTemplate(
    template='Write a detail report on {topic}',
    input_variables=['topic']
)

# 2nd Prompt --> Summary of Prompt
template2 = PromptTemplate(
    template='Write a 5 line summary on following text. \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'Indian Independence'})

print(result)

'''Thus using Parser our steps are dramatically reduced and our \
    code looks more cleaner and easy to understand and short.
'''