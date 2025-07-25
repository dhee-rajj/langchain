from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings 
from langchain.globals import set_debug

set_debug(True)

llm = OllamaEmbeddings(model="llama3.2:latest")

text = input("Enter the text\n")

response = llm.embed_query(text)
print()
print(response)