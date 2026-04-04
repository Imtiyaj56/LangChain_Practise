'''Used to load the information from web using its URL.'''
'''It uses the beautiful soup in background to parse HTML and extract visible text.'''
'''Used with static pages in website.'''
'''
Limitations : 
        - does not handle javascript heavy pages well, for that use SeleniumURLLoader,
        - loads only static pages, not a content which are rendering with time.
'''

from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile" )

parser = StrOutputParser()

url = 'https://www.aljazeera.com/'
loader = WebBaseLoader(url)
doc = loader.load()

template = PromptTemplate(
    template= 'Answer the following \n {question} from following \n {text}.',
    input_variables=['question','text']
)

chain = template | model | parser
result = chain.invoke({'question': 'what about the rescue mission of USA in Iran?', 'text': doc})
print(result)