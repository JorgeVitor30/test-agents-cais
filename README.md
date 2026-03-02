# Agente Conversor de Temperaturas com LangChain + Ollama

Este projeto implementa um Agente de IA utilizando LangChain integrado ao Ollama (modelo Llama 3.2) para realizar conversões de temperatura entre:
- Celsius ↔ Fahrenheit
- Celsius ↔ Kelvin
O agente utiliza ferramentas criadas via código para executar os cálculos de conversão

## Tecnologias Utilizadas

- LangChain
- Python
- Ollama
- Modelo: Llama3.2

## Como Executar o Projeto

### Instalar o Ollama

Para executar o programa é necessário que você tenha o [Ollama](https://ollama.com/download/windows) instalado em sua máquina para que seja possível usar o modelo de linguagem.

Após instalar, baixe o modelo:
```
ollama pull llama3.2
```

### Criar Ambiente Virtual

```
python -m venv venv
venv\Scripts\activate
```

### Instalar Dependências

Execute:
```
pip install -r requirements.txt
```

### Executar o Projeto

```
python main.py
```

### Autora

Beatriz Andrade

Linkedin : https://www.linkedin.com/in/beatriz-andrade-94b38b233/
