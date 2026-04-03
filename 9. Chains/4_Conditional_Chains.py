from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile" )

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['Positive', 'Negative'] = Field(description= 'Give the sentiment of the following feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

template1 = PromptTemplate(
    template= 'Classify the sentiment of the following feedback text in Positive or Negative. \n FeedBack : {feedback} \n {format_instruction}',
    input_variables= ['feedback'],
    partial_variables= {'format_instruction': parser2.get_format_instructions()}
)

classifier_Chain = template1 | model | parser2

template2 = PromptTemplate(
    template= 'Write an appropriate response to this postive feedback. \n {feedback}',
    input_variables= ['feedback']
)

template3 = PromptTemplate(
    template= 'Write an appropriate response to this negative feedback. \n {feedback}',
    input_variables= ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'Positive', template2 | model | parser),
    (lambda x:x.sentiment == 'Negative', template3 | model | parser),
    RunnableLambda(lambda x: 'Could not find any sentiment')
)

chain = classifier_Chain | branch_chain

result = chain.invoke({'feedback': 'This is very good smartphone.'})
print(result)