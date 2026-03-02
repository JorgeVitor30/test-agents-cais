from langchain_core.tools import tool
from regressor_agent import RegressorAgent

_regressor_instance = None

def get_regressor():
    global _regressor_instance
    if _regressor_instance is None:
        _regressor_instance = RegressorAgent()
    return _regressor_instance

@tool
def celsius_to_fahrenheit(value: float) -> float:
    """Converte temperatura de Celsius para Fahrenheit."""
    return (value * 9/5) + 32

@tool
def fahrenheit_to_celsius(value: float) -> float:
    """Converte temperatura de Fahrenheit para Celsius."""
    return (value - 32) * 5/9

@tool
def celsius_to_kelvin(value: float) -> float:
    """Converte temperatura de Celsius para Kelvin."""
    return value + 273.15

@tool
def kelvin_to_celsius(value: float) -> float:
    """Converte temperatura de Kelvin para Celsius."""
    return value - 273.15

@tool
def predict_iris_petal_width(sepal_length: float, sepal_width: float, petal_length: float) -> str:
    """
    Prediz a largura da pétala (petal_width) de uma flor iris baseado em suas características.
    
    Args:
        sepal_length: Comprimento da sépala em cm
        sepal_width: Largura da sépala em cm
        petal_length: Comprimento da pétala em cm
    
    Returns:
        String com a predição formatada
    """
    regressor = get_regressor()
    prediction = regressor.predict(sepal_length, sepal_width, petal_length)
    return f"Predição de largura da pétala: {prediction:.2f} cm"