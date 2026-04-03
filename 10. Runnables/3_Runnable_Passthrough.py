'''It returns the output same as the input, input = output'''
'''Where can be used: In RunnableSequence Example, final output is only
   explanation of joke, not joke, but what if I want joke anlong with
   its explanation, at that RunnablePassthrough is used as placeholder,
   where it directly passed joke as output in one Parallel chain and another
   chain works for explanation of that joke.'''

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile")

template = PromptTemplate(
    template= 'Write a joke about {topic}',
    input_variables= ['topic']
)

template2 = PromptTemplate(
    template= 'Explain the following joke. \n {joke}',
    input_variables= ['joke']
)

parser = StrOutputParser()

joke_generator = RunnableSequence(template, model, parser)

parallel_chain = RunnableParallel({
   'joke': RunnablePassthrough(),
   'explanation': RunnableSequence(template2, model, parser)}
)

final_chain = RunnableSequence(joke_generator, parallel_chain)
result = final_chain.invoke({'topic': 'Khakhi Chaddi Gang'})
print(result)
