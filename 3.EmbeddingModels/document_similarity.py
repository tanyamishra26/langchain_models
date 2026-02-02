from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding= OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)
documents= [
    "The capital of India is New Delhi.",
    "The capital of France is Paris.",
    "The capital of Germany is Berlin."   

]
query= "What is the capital of India?"
doc_embeddings= embedding.embed_documents(documents)
query_embedding= embedding.embed_query(query)

scores= cosine_similarity([query_embedding], doc_embeddings)        # always pass 2D arrays to cosine_similarity function

index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]  # prints the index and similarity score of the most similar document

print(query)
print(documents[index])
print("Similarity Score is :", score)

