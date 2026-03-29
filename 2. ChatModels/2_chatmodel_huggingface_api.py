from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm, temperature=1.2)

result = model.invoke("write an essay on culture of Iran and its peoples with regards on their muslim culture.")

print(result.content)