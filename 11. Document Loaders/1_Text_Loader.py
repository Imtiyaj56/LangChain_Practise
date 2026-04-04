from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("joke.txt", encoding = 'utf-8')
docs = loader.load()

model = ChatGroq(model = "llama-3.3-70b-versatile" )

parser = StrOutputParser()

template = PromptTemplate(
    template= 'generate a quiz for following text. \n {text}',
    input_variables=['text']
)

chain = RunnableSequence(template, model, parser)
print(chain.invoke({'text': docs[0].page_content}))

