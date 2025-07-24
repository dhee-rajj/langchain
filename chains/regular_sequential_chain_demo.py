from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                 base_url="https://models.github.ai/inference",
                model = "openai/gpt-4o")

topic_prompt = PromptTemplate(
    input_variables=["topic"],
    template = """You are an experienced speech writer.
        You need to craft an impactful title for a speech
        on the following topic: {topic}
        Answer exactly with one title.
        """
)

speech_prompt = PromptTemplate(
    input_variables=["title", "emotion"],
    template="""You need to write a powerful speech of 350 words
        for the following title: {title}
        with the emotion {emotion}
        format the output with 2 keys: 'title', 'speech' and fill them
        with the respective values
        """
)

first_chain = topic_prompt | llm | StrOutputParser()
second_chain = speech_prompt | llm | JsonOutputParser()
final_chain = first_chain | (lambda title: {"title": title, "emotion": emotion}) | second_chain

st.title("Speech Generator")

topic = st.text_input("Enter the topic")
emotion = st.text_input("Enter the emotion")

if topic and emotion:
    response = final_chain.invoke({"topic": topic})
    st.write(response)

