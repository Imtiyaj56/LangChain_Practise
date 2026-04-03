from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableSequence

load_dotenv()

template = PromptTemplate(
    template= 'Write a joke about {topic}',
    input_variables= ['topic']
)

template2 = PromptTemplate(
    template= 'Explain the following joke. \n {joke}',
    input_variables= ['joke']
)

parser = StrOutputParser()

model = ChatGroq(model = "llama-3.3-70b-versatile")

chain = RunnableSequence(template, model, parser, template2, model, parser)

result = chain.invoke({'topic': 'Gujarati People'})
print(result)