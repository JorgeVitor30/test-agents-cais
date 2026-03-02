from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np


class RegressorAgent:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train()
        
    def load_and_prepare_data(self):
        iris = load_iris()
        
        X = iris.data[:, :3]  
        y = iris.data[:, 3]
        
        self.feature_names = ['sepal_length', 'sepal_width', 'petal_length']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Dataset carregado: {len(self.X_train)} amostras de treino, {len(self.X_test)} amostras de teste")
        print(f"Features: {', '.join(self.feature_names)}")
        print(f"Target: petal_width\n")
        
    def train(self):
        if self.X_train is None:
            self.load_and_prepare_data()
            
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)
        
        print(f"Modelo treinado com sucesso!")
        print(f"R² no conjunto de treino: {train_score:.4f}")
        print(f"R² no conjunto de teste: {test_score:.4f}\n")
        
        return train_score, test_score
    
    def predict(self, sepal_length, sepal_width, petal_length):
        if self.model is None:
            raise ValueError("Modelo não treinado! Execute o método train() primeiro.")
        
        input_data = np.array([[sepal_length, sepal_width, petal_length]])
        
        prediction = self.model.predict(input_data)[0]
        
        return prediction








# agent = RegressorAgent()

# valores = (50, 40, 90)

# sepal_length = float(valores[0])
# sepal_width = float(valores[1])
# petal_length = float(valores[2])

# prediction = agent.predict(sepal_length, sepal_width, petal_length)

# print(f"\n{'='*60}")
# print(f"Input:")
# print(f"  - Sepal Length: {sepal_length} cm")
# print(f"  - Sepal Width: {sepal_width} cm")
# print(f"  - Petal Length: {petal_length} cm")
# print(f"\nPredição:")
# print(f"  - Petal Width: {prediction:.2f} cm")
# print(f"{'='*60}\n")
    