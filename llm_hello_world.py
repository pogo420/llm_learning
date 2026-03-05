from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# model initialization
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.5,  # reducing randomness
    top_k=1  # reducing randomness
    )

# Reuseable prompt with parameterizarion
independece_question = PromptTemplate.from_template("{country} became independent in which year?(No preamble)")

# Chaining, based classes might have implemented __or__ methods.
chain = independece_question | llm

# Getting responses
for country in ["India", "Nepal", "South Africa"]:
    response = chain.invoke({"country": country})
    print(response.content)

print("-------------------------"*5)

# Example on potential of prompt.
independece_question_v2 = PromptTemplate.from_template(
    """Give me year of independence for list of countries in json format.
    key will be country and yearOfIndependence

    Example:
        country: china, yearOfIndependence: 1950

    Output will be list of json. No preamble.

    List of countries
    -------
    {countries}
    """
)

chain2 = independece_question_v2 | llm
response2 = chain2.invoke({"countries": ["Bangladesh", "Nepal", "South Africa"]})
print(JsonOutputParser().parse(response2.content))
