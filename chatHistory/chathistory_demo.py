from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_41")

llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                 base_url="https://models.github.ai/inference",
                model = "openai/gpt-4.1")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a Devops Coach. Answer any questions "
                "related to devops, answer in less than 100 words"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt_template | llm

history_for_chain = ChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id:history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)

print("Devops Guide")

while True:
    question = input("Enter the question")
    if question:
        response = chain_with_history.invoke({"input":question}, {
                "configurable":{
                "session_id":"abc123"
            }
        })
        print(response.content)
