from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="text-embedding-004", dimensions=32)

documents = [
    "Delhi is capital of India",
    'Tehran is Capital of Iran',
    'Dublin is capital of Ireland'
]

result = embedding.embed_documents(documents)

print(str(result))