'''Two Retrievers which we have seen earlier are based on from where we are 
   retrieving the data, but this MMR retriever is based on how the data is 
   retrieved from any data source.'''

'''MMR : Maximal Marginal Relevance'''

'''It fetch the results which are relevant to query also diversed among each other
   not same. Bcz fetching all docs of same information is waste of space, so we want 
   more diversified fetched result or object.'''

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

#embedding model
embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

#creating vector store (FAISS)
vectorstore = FAISS.from_documents(
    documents= docs,
    embedding= embedding_model
)

#Enabling MMR as retriever
retriever = vectorstore.as_retriever(
    search_type = 'mmr',
    search_kwargs = {'k': 3, 'lambda_mult': 0.5}  #k = no of relevant docs, lambda_mult = relevance diversity balance
)

query = 'What is Langchain ?'
results = retriever.invoke(query)

print(results)