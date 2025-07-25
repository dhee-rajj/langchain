from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.globals import set_debug


set_debug(True)


llm = OllamaEmbeddings(model="llama3.2:latest")

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
