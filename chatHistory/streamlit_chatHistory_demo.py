from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
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
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt_template | llm

history_for_chain = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id:history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)


st.title("Devops Guide")

question = st.text_input("Enter the question")

if question:
    response = chain_with_history.invoke({"input":question}, {
            "configurable":{
            "session_id":"abc123"
        }
    })
    st.write(response.content)
    # st.write(history_for_chain)
