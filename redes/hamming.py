import numpy as np

class RedHamming:
    def __init__(self, num_neuronas, num_patrones):
        self.N = num_neuronas
        self.M = num_patrones
        self.W = np.zeros((self.M, self.N))
        self.bias = self.N / 2
        self.epsilon = 1 / (self.M - 1) if self.M > 1 else 0

    def funcion_transferencia(self, valor):
        return np.maximum(0, valor)

    def entrenar(self, patrones):
        for i in range(self.M):
            self.W[i] = np.array(patrones[i]) / 2

    def predecir(self, patron, max_iter=1000):
        # Fase 1: Similitud
        similitud = np.dot(self.W, patron) + self.bias
        a = self.funcion_transferencia(similitud * (1 / self.N))
        
        # Fase 2: MAXNET
        historial = [a.copy()]
        activaciones = a.copy()
        
        for it in range(1, max_iter + 1):
            nuevas_activaciones = np.zeros_like(activaciones)
            suma_total = np.sum(activaciones)
            
            for i in range(self.M):
                suma_otros = suma_total - activaciones[i]
                resultado = activaciones[i] - (self.epsilon * suma_otros)
                nuevas_activaciones[i] = self.funcion_transferencia(resultado)
            
            activaciones = nuevas_activaciones.copy()
            historial.append(activaciones.copy())
            
            # Condición de paro: queda 1 o 0 neuronas vivas
            activos = np.count_nonzero(activaciones)
            if activos <= 1:
                break
                
        if np.count_nonzero(activaciones) == 1:
            ganador = int(np.argmax(activaciones))
            return "CONVERGE", ganador, it, historial
        else:
            return "NULO", -1, it, historial