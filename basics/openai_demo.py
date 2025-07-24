from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug
import os

load_dotenv(dotenv_path="../.env")

set_debug(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                 base_url="https://models.github.ai/inference",
                model = "openai/gpt-4o")

question = input("Enter the question")

response = llm.invoke(question)

print(response.content)