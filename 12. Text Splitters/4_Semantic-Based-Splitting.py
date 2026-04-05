'''It is used to split the text or document based on semantic meaning of chunk.'''
'''It is still in experiment, thus results are not that accurate.'''
'''We need to used embedding model, to convert text into embedding to capture
   semantic meaning using cosine similarity score between to sentences. '''

from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="text-embedding-004", dimensions=32)

splitter = SemanticChunker(
    embeddings= embedding, breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold_amount=1
)

text = '''The rapid development of artificial intelligence has revolutionized the technology sector. Machine learning models, particularly large language models, can now generate human-like text, analyze complex datasets, and automate coding tasks. Many companies are integrating AI to improve operational efficiency and create new products.

Meanwhile, the global climate is undergoing significant changes due to increased greenhouse gas emissions. Scientists warn that rising temperatures are leading to melting glaciers, sea-level rise, and more frequent extreme weather events. Urgent action is needed to transition to renewable energy sources like solar and wind power.

In contrast, the field of marine biology is exploring the deep ocean, discovering new species in hydrothermal vents. These deep-sea organisms live in complete darkness and high pressure. Researchers are fascinated by their unique adaptations, which could provide insights into the origins of life on Earth.
'''

docs = splitter.create_documents([text])
print(len(docs))
print(docs[2])
