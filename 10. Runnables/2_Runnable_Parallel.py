from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableParallel, RunnableSequence

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile")

template1 = PromptTemplate(
    template= 'You are professional Linkdin Expert, generate the Linkdin post about {topic}',
    input_variables= ['topic']
)

template2 = PromptTemplate(
    template= 'You are professional tweeter Expert, generate the tweet about {topic}',
    input_variables= ['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(template1, model, parser),
    'linkdin': RunnableSequence(template2, model, parser)
}
)

print(parallel_chain.invoke({'topic': 'Langchain, The Impacting Technology'}))