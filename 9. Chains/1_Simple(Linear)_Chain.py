from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile" )

template = PromptTemplate(
    template= 'Imagine you are teacher who teach students in simple language, Give 5 interesting facts about {topic}',
    input_variables= ['topic']
)

parser = StrOutputParser()

chain = template | model | parser
result = chain.invoke({'topic': 'Cow'})
print(result)

