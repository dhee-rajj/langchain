from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                 base_url="https://models.github.ai/inference",
                model = "openai/gpt-4o")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a Devops Coach. Answer any questions "
                "related to devops"),
    ("human", "{input}")
])

chain = prompt_template | llm

st.title("Devops Guide")

question = st.text_input("Enter the question")

if question:
    response = chain.invoke({"input":question})
    st.write(response.content)