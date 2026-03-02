from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from tools import (
    celsius_to_fahrenheit, 
    fahrenheit_to_celsius, 
    celsius_to_kelvin, 
    kelvin_to_celsius,
    predict_iris_petal_width
)
from langchain.agents import create_agent

llm = ChatOllama(model="llama3.2")

template = """
    Você é um agente de IA versátil com duas especialidades:
    
    1. CONVERSÃO DE TEMPERATURA: Converter unidades de temperatura (Celsius, Fahrenheit, Kelvin)
       - Identificar a unidade de temperatura fornecida
       - Converter para a unidade solicitada
       - Fornecer a resposta no formato: "<valor> <unidade>"
       - Se a unidade não for reconhecida, informe o usuário
    
    2. PREDIÇÃO DE CARACTERÍSTICAS DE FLOR IRIS: Prever a largura da pétala
       - Se o usuário pedir para prever, estimar ou calcular a largura da pétala de uma flor iris
       - Use a ferramenta predict_iris_petal_width com os valores de sepal_length, sepal_width e petal_length
       - Responda de forma clara com o resultado da predição
    
    Você deve:
        - Usar a ferramenta correta baseado no contexto da pergunta
        - Responder de forma amigável e clara
        - Se não conseguir identificar qual ferramenta usar, peça esclarecimentos

    Com base em suas ferramentas, responda a seguinte pergunta: {input}

"""


tools = [
    celsius_to_fahrenheit,
    fahrenheit_to_celsius,
    celsius_to_kelvin,
    kelvin_to_celsius,
    predict_iris_petal_width,
]

agent = create_agent(model=llm, tools=tools, system_prompt=template)


while True:
    question = input("Faça uma pergunta: ")
    if question == "sair":
        break

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    final_response = result["messages"][-1].content

    print(f"Resposta: {final_response}")
