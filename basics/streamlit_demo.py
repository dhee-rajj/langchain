import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.globals import set_debug

set_debug(True)

llm = ChatOllama(model="mistral:7b-instruct-q4_K_M")

st.title("Ask Anything")

question = st.text_input("Enter the question")

if question:
    response = llm.invoke(question)
    st.write(response.content)