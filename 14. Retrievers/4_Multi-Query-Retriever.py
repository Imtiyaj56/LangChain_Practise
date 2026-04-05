'''What is a query provided by user contain more than one meaning.'''
'''At that time our retriever should be capable of entertaining all aspects
   of query and retrieve relevant information for accurate answers.'''

'''Ex : How can I stay Healthy ?
        --> What should I eat?
        --> How much should I exercise?
        --> How much should I sleep?
'''

'''Above single query has different meaning, and thus our retriever should be 
   capable of retrieving relevant info to all meaning.'''

''' Query --> LLM --> Multiple Queries --> Retrievers --> All Relevant Docs'''

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

#embedding model
embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

#creating vector store (FAISS)
vectorstore = FAISS.from_documents(
    documents= all_docs,
    embedding= embedding_model
)

#Creating Multiquery Retriever
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatGroq(model="llama-3.3-70b-versatile")
)

# Query
query = "How to improve energy levels and maintain balance?"

# Retrieve results
multiquery_results= multiquery_retriever.invoke(query)
print(multiquery_results)