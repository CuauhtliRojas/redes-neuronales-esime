import numpy as np

class RedHopfield:
    def __init__(self, tamano, tipo_func='bipolar'):
        self.tamano = tamano
        self.tipo_func = tipo_func
        self.W = np.zeros((tamano, tamano))

    def _convertir_interno(self, vector):
        if self.tipo_func == 'binario':
            return np.where(vector <= 0, -1, 1)
        return vector.copy()

    def _convertir_salida(self, vector):
        if self.tipo_func == 'binario':
            return np.where(vector == -1, 0, 1)
        return vector

    def entrenar(self, patrones):
        desglose_formulas = []
        for i in range(self.tamano):
            for j in range(i + 1, self.tamano):
                suma_val = 0
                terminos = []
                for p in patrones:
                    p_int = self._convertir_interno(p)
                    val = p_int[i] * p_int[j]
                    suma_val += val
                    terminos.append(f"({p_int[i]})({p_int[j]})")
                
                self.W[i, j] = suma_val
                self.W[j, i] = suma_val
                
                formula_str = f"w_{i+1},{j+1} = {' + '.join(terminos)} = {suma_val}  ->  [Reflejo: w_{j+1},{i+1} = {suma_val}]"
                desglose_formulas.append(formula_str)
                
        np.fill_diagonal(self.W, 0)
        return desglose_formulas

    def calcular_hamming(self, vec1, vec2):
        return np.sum(vec1 != vec2)

    def predecir(self, vector_prueba, max_iter=50):
        estado_actual = self._convertir_interno(vector_prueba)
        vector_original = estado_actual.copy()
        historial = []
        estados_visitados = []

        for iteracion in range(1, max_iter + 1):
            estado_anterior = estado_actual.copy()
            estados_visitados.append(estado_anterior.tolist())

            # Calculamos 'y' para todas las neuronas a la vez
            # usando únicamente el estado_anterior
            y_total = np.dot(self.W, estado_anterior)

            for neurona in range(self.tamano):
                # Evaluamos basándonos en el y_total calculado previamente
                if y_total[neurona] > 0:
                    estado_actual[neurona] = 1
                elif y_total[neurona] < 0:
                    estado_actual[neurona] = -1
                # Si es 0, no cambia

            distancia = self.calcular_hamming(vector_original, estado_actual)
            historial.append(
                (
                    iteracion,
                    self._convertir_salida(estado_actual.copy()),
                    distancia
                )
            )

            # Checar convergencia
            if np.array_equal(estado_actual, estado_anterior):
                estado_final = self._convertir_salida(estado_actual)
                dist_final = self.calcular_hamming(vector_original, estado_actual)
                return "converge", estado_final, iteracion, dist_final, historial

            # Checar bucle (ahora sí saltará correctamente)
            if estado_actual.tolist() in estados_visitados:
                estado_final = self._convertir_salida(estado_actual)
                dist_final = self.calcular_hamming(vector_original, estado_actual)
                return "bucle", estado_final, iteracion, dist_final, historial

        return "limite", self._convertir_salida(estado_actual), max_iter, self.calcular_hamming(vector_original, estado_actual), historial
    def encontrar_patron_mas_cercano(self, estado_final, patrones):
        distancias = []

        for idx, p in enumerate(patrones):
            p_int = self._convertir_interno(p)
            distancia = self.calcular_hamming(estado_final, p_int)
            distancias.append((idx + 1, distancia))

        distancias.sort(key=lambda x: x[1])
        return distancias