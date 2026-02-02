# generating embedding for multiple documents (query) 
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "The capital of India is New Delhi.",
    "The capital of France is Paris.",
    "The capital of Germany is Berlin."
]
result = embeddings.embed_documents(documents)                # embed_ documents-->generates embeddings for multiple documents

print(str(result))
