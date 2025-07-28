from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.globals import set_debug
import os

load_dotenv(dotenv_path="../.env")

# set_debug(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_41")

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, 
                 base_url="https://models.github.ai/inference",
                model = "openai/text-embedding-3-small")
llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                 base_url="https://models.github.ai/inference",
                model = "openai/gpt-4o")


document = TextLoader("product-data.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                               chunk_overlap=200)
chunks = text_splitter.split_documents(document)
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant for answering questions.
                Use the provided context to respond.If the answer
                isn't clear, acknowledge that you don't know.
                Limit your response to three concise sentences.
                {context}
                """
     ),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, qa_chain)

print("chat with documents")
while True:
    question = input("Enter the question: ")

    if question:
        response = rag_chain.invoke({"input": question })
        print(response['answer'])