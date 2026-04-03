'''To make conditional workflows, it provide the functionality of if else statement.'''

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableBranch, RunnableLambda, RunnableParallel, RunnableSequence, RunnablePassthrough

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile")

template = PromptTemplate(
    template= 'Write a report about {topic}',
    input_variables= ['topic']
)

template2 = PromptTemplate(
    template= 'Write an appropriate summary for this report. \n Report : {report}',
    input_variables= ['report']
)

def word_count(text):
    return len(text.split())

parser = StrOutputParser()

report_generator_chain = RunnableSequence(template, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(template2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_generator_chain, branch_chain)
result = final_chain.invoke({'topic': 'Afternoon of Summer in Typical Indian Village'})
print(result)






