from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.globals import set_debug
import os

load_dotenv(dotenv_path="../.env")

set_debug(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_41")

llm = OpenAIEmbeddings(api_key=OPENAI_API_KEY, 
                 base_url="https://models.github.ai/inference",
                model = "openai/text-embedding-3-small")

document = TextLoader("job_listings.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks = text_splitter.split_documents(document)
db = Chroma.from_documents(chunks, llm)
retriever = db.as_retriever()

text = input("Enter the text")
# embedding_vector = llm.embed_query(text)

# docs = db.similarity_search_by_vector(embedding_vector)
docs = retriever.invoke(text)

for doc in docs:
    print(doc.page_content)
