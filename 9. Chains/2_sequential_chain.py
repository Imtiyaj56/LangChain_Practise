from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

template1 = PromptTemplate(
    template= '''Imagine you are {role} expert writer, your work is to write a detailed report
                 on {topic}.''',
    input_variables= ['role', 'topic']
)

template2 = PromptTemplate(
    template= '''Imagine you are expert summary writer, your work is to find 5 most 
                 important points from following {report}''',
    input_variables= ['report']
)

model1 = ChatGroq(model = "llama-3.3-70b-versatile" )

llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Meta-Llama-3-8B-Instruct',
    task = 'text-generation'
)

model2 = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = template1 | model1 | parser | template2 | model2 | parser
result = chain.invoke({'role': 'defence strategy', 'topic': 'hypersonic cruise missile'})
print(result)






