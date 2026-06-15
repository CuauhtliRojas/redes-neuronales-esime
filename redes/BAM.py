import numpy as np

class RedBAM:
    def __init__(self, di, do):
        self.di = di
        self.do = do
        self.W = np.zeros((di, do), dtype=float)

    def entrenar(self, patrones_input, patrones_output):
        """Calcula la matriz de pesos asociando los pares (X, Y)."""
        desglose_formulas = []
        pp = len(patrones_input)
        
        for z in range(self.di):
            for j in range(self.do):
                val = 0
                terminos = []
                for a in range(pp):
                    x_val = patrones_input[a][z]
                    y_val = patrones_output[a][j]
                    val += (x_val * y_val)
                    terminos.append(f"({x_val})({y_val})")
                
                self.W[z, j] = val
                formula_str = f"w_{z+1},{j+1} = {' + '.join(terminos)} = {val}"
                desglose_formulas.append(formula_str)
                
        return desglose_formulas

    def _funcion_transferencia(self, vector):
        """Aplica la función escalón bipolar."""
        return np.where(vector >= 0, 1, -1)

    def predecir_ida(self, vector_x):
        """Operación X -> Y"""
        y_crudo = self.W.T @ vector_x
        return self._funcion_transferencia(y_crudo)

    def predecir_vuelta(self, vector_y):
        """Operación Y -> X"""
        x_crudo = self.W @ vector_y
        return self._funcion_transferencia(x_crudo)