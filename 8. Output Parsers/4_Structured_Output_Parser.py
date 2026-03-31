''' 
    This parser is used to parse the output of LLM in JSON form along with the
    specified schema(using ResponseSchema() method), thus it solve the problem of Json Output Parser.
'''

from langchain_classic.output_parsers.structured import StructuredOutputParser, ResponseSchema
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Meta-Llama-3-8B-Instruct',
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template= 'Give 3 facts about the {topic}. \n {format_instruction}',
    input_variables= ['topic'],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'topic': 'Pearl Harbour Attack'})
print(result)

"""
    Disadvantge : Data Validation is not present in this, ex : age -> int(defined), 
    but age -> str(also accept).

    Soluion : Pydantic Output Parser
"""