import numpy as np
from rich.console import Console

# Importamos tu motor matemático
from redes.hopfield import RedHopfield

console = Console(color_system=None)  # Cero colores, formato consola pura

def generar_vectores_limpios(cantidad, tamano):
    """Genera vectores bipolares aleatorios."""
    return [np.random.choice([-1, 1], size=tamano) for _ in range(cantidad)]

def aplicar_ruido(vector, porcentaje):
    """Invierte el signo del % de los bits especificados para crear ruido."""
    vec_ruido = vector.copy()
    num_cambios = int(len(vector) * (porcentaje / 100.0))
    indices = np.random.choice(len(vector), num_cambios, replace=False)
    for idx in indices:
        vec_ruido[idx] *= -1 
    return vec_ruido

def main():
    tamano = 100 # Necesitamos al menos 100 para que los porcentajes sean exactos
    console.print("================================================================")
    console.print("REPORTE DE ENTRENAMIENTO DE HOPFIELD CON 30 VECTORES SIMULTÁNEOS")
    console.print("================================================================\n")

    # 1. Generación exacta de lo que pidió la maestra
    console.print("1. GENERACIÓN DE VECTORES DE ENTRENAMIENTO...")
    limpios = generar_vectores_limpios(10, tamano)
    ruido_10 = [aplicar_ruido(v, 10) for v in limpios]
    ruido_35 = [aplicar_ruido(v, 35) for v in limpios]

    todos_los_vectores = limpios + ruido_10 + ruido_35

    # 2. Imprimir los vectores para el reporte
    console.print("\n[LISTADO DE VECTORES A MEMORIZAR]")
    for i, vec in enumerate(todos_los_vectores):
        if i == 0:
            console.print("\n--- 10 VECTORES LIMPIOS ---")
        elif i == 10:
            console.print("\n--- 10 VECTORES CON 10% DE RUIDO ---")
        elif i == 20:
            console.print("\n--- 10 VECTORES CON 35% DE RUIDO ---")
            
        vec_str = " ".join(map(str, vec))
        console.print(f"V_{i+1}: {vec_str}")

    # 3. Entrenamiento con los 30 vectores (El colapso)
    console.print("\n2. ENTRENANDO LA RED...")
    red = RedHopfield(tamano=tamano, tipo_func='bipolar')
    red.entrenar(todos_los_vectores)
    console.print(f"Red entrenada exitosamente. Matriz de pesos W calculada ({tamano}x{tamano}).")

    # 4. Demostración del fracaso
    console.print("\n3. FASE DE PRUEBA (Comprobación de la matriz)")
    console.print("Evaluando el 'Vector Limpio 1' para ver si la red puede recordarlo...\n")
    
    estado_red, estado_final, iteraciones, dist_final, historial = red.predecir(limpios[0])

    console.print(f"Entrada (V_1 Limpio): {' '.join(map(str, limpios[0]))}")
    console.print(f"Salida de la red:     {' '.join(map(str, estado_final))}")
    console.print(f"Bits modificados erróneamente por la red: {dist_final} de {tamano}")

    if np.array_equal(estado_final, limpios[0]):
        console.print("Resultado: LOGRÓ RECUPERARLO (Estadísticamente casi imposible en este escenario)")
    else:
        console.print("Resultado: FRACASO. La red arrojó un estado espurio irreconocible.")

    # 5. Justificación Teórica
    console.print("\n" + "="*80)
    console.print("ANÁLISIS Y CONCLUSIÓN PARA EL REPORTE:")
    console.print("Al ejecutar el entrenamiento con 30 vectores simultáneos en una matriz de 100x100,")
    console.print("se comprueba el colapso matemático de la memoria asociativa por dos motivos:")
    console.print("\n1. Sobrecarga de Capacidad: La capacidad máxima teórica de Hopfield es P_max = 0.138 * N.")
    console.print("   Para N=100, el límite es de 13.8 patrones. Al ingresar 30, se superó la capacidad en un 217%.")
    console.print("\n2. Interferencia Destructiva (Pesos Contradictorios): Al alimentar la matriz con")
    console.print("   versiones ruidosas de los mismos vectores durante el aprendizaje, la Regla de Hebb")
    console.print("   memoriza el ruido como información válida. Esto destruye los atractores originales")
    console.print("   y genera estados espurios, imposibilitando la convergencia hacia un patrón útil.")
    console.print("="*80 + "\n")

if __name__ == "__main__":
    main()