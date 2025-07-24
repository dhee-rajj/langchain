from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

openai_llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                 base_url="https://models.github.ai/inference",
                 model = "openai/gpt-4o"
                )

xai_llm = ChatOpenAI(api_key=XAI_API_KEY, 
                 base_url="https://models.github.ai/inference",
                 model = "xai/grok-3"
                )

topic_prompt = PromptTemplate(
    input_variables=["topic"],
    template = """You are an experienced speech writer.
        You need to craft an impactful title for a speech
        on the following topic: {topic}
        Answer exactly with one title.
        """
)

speech_prompt = PromptTemplate(
    input_variables=["title"],
    template="""You need to write a powerful speech of 350 words
        for the following title: {title}
        """
)

first_chain = topic_prompt | openai_llm | StrOutputParser() #| (lambda title: (st.write(title), title)[1])
second_chain = first_chain | xai_llm
final_chain = first_chain | second_chain

st.title("Speech Generator")

topic = st.text_input("Enter the topic")

if topic:
    response = final_chain.invoke({"topic": topic})
    st.write(response.content)

