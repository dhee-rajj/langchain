from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                 base_url="https://models.github.ai/inference",
                model = "openai/gpt-4o")
prompt_template = PromptTemplate(
    input_variables=["country", "no_of_paras", "language"],
    template="""You are an expert in traditional cuisines.You provide 
        information about a specific dish from a specific country
        Avoid giving information about fictional places. 
        If the country is fictional or non-existent answer: I don't know.
        Answer the question: What is the traditional cuisine of {country}.
        Answer in {no_of_paras} short paragraphs in {language} language.
    """
)

st.title("Cusine Finder")

country = st.text_input("Enter the country")
no_of_paras = st.number_input("Enter no of paras",min_value=1, max_value=5)
language = st.text_input("Enter the language")

if country and no_of_paras and country:
    response = llm.invoke(prompt_template.format(country=country, 
                            no_of_paras=no_of_paras,
                            language=language))
    st.write(response.content)
