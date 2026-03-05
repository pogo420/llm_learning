from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# model initialization
llm = ChatOllama(model="llama3.2:1b", temperature="1.8")

# Reuseable prompt with parameterizarion
independece_question = PromptTemplate.from_template("{country} became independent in which year?")

# Chaining, based classes might have implemented __or__ methods.
chain = independece_question | llm

# Getting responses
for country in ["India", "Nepal", "South Africa"]:
    response = chain.invoke({"country": country})
    print(response)
