from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                 base_url="https://models.github.ai/inference",
                model = "openai/gpt-4o")

prompt_template = PromptTemplate(
    input_variables=["city", "month", "language", "budget"],
    template="""Welcome to the {city} travel guide!
            If you're visiting in {month}, here's what you can do:
            1. Must-visit attractions.
            2. Local cuisine you must try.
            3. Useful phrases in {language}.
            4. Tips for traveling on a {budget} budget.
            Enjoy your trip!
        """
)

st.title("City Guide")

city = st.text_input("Enter the city name")
month = st.text_input("Enter the month name")
language = st.text_input("Enter the languge")
budget = st.selectbox("Select the budget", ["low", "medium", "high"])

if city and month and language and budget:
    response = llm.invoke(prompt_template.format(city=city,
                                                 month=month,
                                                 language=language,
                                                 budget=budget))
    
    st.write(response.content)