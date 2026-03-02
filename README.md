# Agente com Ferramentas de Conversão e Regressão com LangChain + Ollama

Este projeto implementa um Agente de IA utilizando LangChain integrado ao Ollama (modelo Llama 3.2) para:
- Celsius ↔ Fahrenheit
- Celsius ↔ Kelvin
- Predição de largura da pétala (`petal_width`) no dataset Iris por regressão linear

O agente utiliza ferramentas criadas via código para executar cálculos e predições.

## Ferramentas Disponíveis

- `celsius_to_fahrenheit(value: float) -> float`
- `fahrenheit_to_celsius(value: float) -> float`
- `celsius_to_kelvin(value: float) -> float`
- `kelvin_to_celsius(value: float) -> float`
- `predict_iris_petal_width(sepal_length: float, sepal_width: float, petal_length: float) -> str`

### Nova função: `predict_iris_petal_width`

Essa função foi adicionada em `tools.py` e utiliza o `RegressorAgent` definido em `regressor_agent.py`.

Entrada:
- `sepal_length` (cm)
- `sepal_width` (cm)
- `petal_length` (cm)

Saída:
- String formatada com a predição de `petal_width` em cm.

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

### Autores

Beatriz Andrade
Jorge Vitor

Linkedin : https://www.linkedin.com/in/beatriz-andrade-94b38b233/
