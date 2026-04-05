from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang='en')

query = 'USA-IRAN Tensions 2026'

docs = retriever.invoke(query)

print(docs[1].page_content)