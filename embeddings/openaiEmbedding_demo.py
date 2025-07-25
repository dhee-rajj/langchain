from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.globals import set_debug
import os

load_dotenv(dotenv_path="../.env")

set_debug(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_41")

llm = OpenAIEmbeddings(api_key=OPENAI_API_KEY, 
                 base_url="https://models.github.ai/inference",
                model = "openai/text-embedding-3-small")

text = input("Enter the text\n")

response = llm.embed_query(text)
print()
print(response)