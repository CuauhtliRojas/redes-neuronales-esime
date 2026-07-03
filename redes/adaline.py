import numpy as np

class Adaline:
    def __init__(self, num_entradas, tasa_aprendizaje=0.1, pesos_iniciales=None, bias_inicial=0.0):
        if pesos_iniciales is not None:
            self.W = np.array(pesos_iniciales, dtype=float)
        else:
            self.W = np.zeros(num_entradas)
            
        self.b = float(bias_inicial)
        self.lr = tasa_aprendizaje

    def predecir_lineal(self, x):
        """Retorna la salida cruda (y = W*x + b) sin activar"""
        return np.dot(self.W, x) + self.b

    def clasificar(self, x):
        """Aplica la función escalón para dar un resultado final bipolar"""
        y_lineal = self.predecir_lineal(x)
        return 1 if y_lineal >= 0 else -1

    def entrenar(self, X, Y, epocas=100, error_minimo=0.01):
        historial = []
        num_patrones = len(X)
        
        for epoca in range(epocas):
            errores_cuadraticos = []
            
            for x, y_deseada in zip(X, Y):
                # 1. Salida cruda (lineal)
                y_pred = self.predecir_lineal(x)
                
                # 2. Error continuo
                error = y_deseada - y_pred
                errores_cuadraticos.append(error ** 2)
                
                # 3. Regla Delta de Widrow-Hoff
                self.W += self.lr * error * x
                self.b += self.lr * error
                
            # 4. Cálculo del Error Cuadrático Medio (ECM o MSE) de la época
            mse = np.sum(errores_cuadraticos) / num_patrones
            historial.append((epoca + 1, self.W.copy(), self.b, mse))
            
            # Condición de paro: Si el MSE es menor al umbral deseado
            if mse <= error_minimo:
                break
                
        return historial