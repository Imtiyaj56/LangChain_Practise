"""Output parsers are used with the models which are not capable of giving
    structured output, ex: mostly open sources model in hugging face."""

"""Output parser can also be used with models which are capable of giving 
    structured output, even they are capable."""

"""Types(which we gonna see):
        (i) String Output Parser
        (ii) Json Output Parser
        (iii) Structured Output Parser
        (iii) Pydantic Output Parser."""

from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Meta-Llama-3-8B-Instruct',
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

# 1st Prompt --> Detailed Prompt
template1 = PromptTemplate(
    template='Write a detail report on {topic}',
    input_variables=['topic']
)

# 2nd Prompt --> Summary of Prompt
template2 = PromptTemplate(
    template='Write a 5 line summary on following text. /n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic': 'Indian Independence'})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result1.content})
result2 = model.invoke(prompt2)
print(result2.content)

