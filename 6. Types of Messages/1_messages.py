from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

messages = [
    SystemMessage(content='You are helpful and knowledgable geopolitical Expert'),
    HumanMessage(content='Tell me about USA vs Vietnam War in short in 7 timeline points.'),
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)