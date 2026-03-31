"""
    It is also Structured Output parser which enforce schema representation along with data 
    validation, thus it solve problem of both "JsonOutputParser" and "StructuredOutputParser".

    Uses Pydantic models.
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Meta-Llama-3-8B-Instruct',
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

#Schema 
class Person(BaseModel):

    name : str = Field('Name of the person')
    age : int = Field('Age Of the Person')
    city : str = Field('City of the person')
    hobbies : list[str] = Field('List of hobbies of person')
    education : Literal['Primary', 'Pre-Primary', 'Uneducated', 'Secondary', 'Higher Secondary', 'Bachelors', 'Masters', 'Phd'] = Field(description='Education level of person')
    criminal_case : Optional[Literal['Yes', 'No']] = Field(default=None, description='Whether the person is engaged in criminal activity or not')


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template= 'Specify the following information of {person}. \n {format_instruction}',
    input_variables= ['person'],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'person': 'Saddam Hossein'})
print(result.model_dump_json())
