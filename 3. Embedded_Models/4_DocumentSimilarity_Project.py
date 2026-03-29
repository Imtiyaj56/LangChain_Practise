from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

document = [
    'Delhi is the capital and biggest city of India, famous for its culture and specially for mughal era.',
    'Mumbai is financial capital of India and second largest city of India, and also known as dream city of India.',
    'Bangalore is the technological capital of India, known for its electronics and IT centres.',
    'Tehran is the capital city of Iran with equal size of Mumbai, and known for its rich culture.',
    'Mashhad is the holiest city for Shia Muslims in Iran and known for its beauty and nature.',
    'Shiraz is the city of poets and philosopher in Iran, it is well known for its contribution to poems and philosophy.'
]

query = 'which city has equal size as of mumbai ?'

doc_embedding = embedding.embed_documents(document)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0]  # all params should be in 2d list

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(document[index])
print('Similarity Score is: ', score)
