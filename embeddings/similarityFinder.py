from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.globals import set_debug
import numpy as np
import os

load_dotenv(dotenv_path="../.env")

set_debug(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_41")

llm = OpenAIEmbeddings(api_key=OPENAI_API_KEY, 
                 base_url="https://models.github.ai/inference",
                model = "openai/text-embedding-3-small")

text1 = input("Enter the text1\n")
text2 = input("Enter the text2\n")

response1 = llm.embed_query(text1)
response2 = llm.embed_query(text2)
print()
similarity_score = np.dot(response1, response2)
print(similarity_score*100,"%")