from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

#source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

#embedding model
embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

#Chroma vector store
vector_store = Chroma.from_documents(
    documents= documents,
    embedding= embedding_model,
    collection_name='my_collection'
)

#convert vector store into retriever
retriever = vector_store.as_retriever(search_kwargs = {'k': 3})

#Query
query = 'What is chroma used for ?'
result = retriever.invoke(query)

print(result.page_content)
