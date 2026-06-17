import numpy as np

class PerceptronSimple:
    def __init__(self, num_entradas, tasa_aprendizaje=0.1, tipo_func="bipolar"):
        self.W = np.zeros(num_entradas)
        self.b = 0.0
        self.lr = tasa_aprendizaje
        self.tipo_func = tipo_func

    def activacion(self, x):
        if self.tipo_func == "bipolar":
            return 1 if x >= 0 else -1
        else: # binario
            return 1 if x >= 0 else 0

    def predecir(self, x):
        suma = np.dot(self.W, x) + self.b
        return self.activacion(suma)

    def entrenar(self, X, Y, epocas=100):
        historial = []
        for epoca in range(epocas):
            errores = 0
            for x, y_deseada in zip(X, Y):
                y_pred = self.predecir(x)
                error = y_deseada - y_pred
                
                if error != 0:
                    # Regla de aprendizaje (Regla Delta)
                    self.W += self.lr * error * x
                    self.b += self.lr * error
                    errores += 1
                    
            historial.append((epoca + 1, self.W.copy(), self.b, errores))
            
            if errores == 0:
                break
        return historial