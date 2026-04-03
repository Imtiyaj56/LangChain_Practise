'''It allows to add custom python function in AI Pipeline'''
'''Acts as middleware between different AI components, enabling preprocessing,
   transformation, API Calls, filtering, post-processing, etc..'''
'''In short, it is used to add anything in pipeline using custom edition.'''
'''Imagine it as custom code node in n8n.'''

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableLambda, RunnableParallel, RunnableSequence, RunnablePassthrough

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile")

template = PromptTemplate(
    template= 'Write a joke about {topic}',
    input_variables= ['topic']
)

def word_count(text):
    return len(text.split())

parser = StrOutputParser()

joke_generator = RunnableSequence(template, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'Words_in_Joke': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_generator, parallel_chain)
result = final_chain.invoke({'topic': 'Chai Vala Prime Minister'})
print(result)

