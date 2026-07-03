import numpy as np

class PerceptronSimple:
    def __init__(self, num_entradas, tasa_aprendizaje=0.1, pesos_iniciales=None, bias_inicial=0.0, tipo_func="bipolar"):
        # Si el usuario mandó pesos, los usamos. Si no, los hacemos ceros (o random si prefieres)
        if pesos_iniciales is not None:
            self.W = np.array(pesos_iniciales, dtype=float)
        else:
            self.W = np.zeros(num_entradas)
            # Nota: Si tu maestra exige random, cambia la línea de arriba por:
            # self.W = np.random.uniform(-1, 1, num_entradas)
            
        self.b = float(bias_inicial)
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
                    # Regla Delta: W_nuevo = W_viejo + (alpha * error * X)
                    self.W += self.lr * error * x
                    # Actualización del bias separada (Matemáticamente equivalente al vector aumentado)
                    self.b += self.lr * error
                    errores += 1
                    
            historial.append((epoca + 1, self.W.copy(), self.b, errores))
            
            # Condición de paro: Si dio 0 errores en esta época, ya aprendió.
            if errores == 0:
                break
        return historial