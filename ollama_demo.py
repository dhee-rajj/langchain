from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="mistral:7b-instruct-q4_K_M")

question = input("Enter the question: ")
response = llm.invoke(question)
print(response.content)
