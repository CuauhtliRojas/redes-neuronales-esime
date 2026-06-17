import numpy as np
import sys
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from redes.hamming import RedHamming

console = Console(color_system=None)
np.set_printoptions(threshold=sys.maxsize, linewidth=250)

def cargar_mosaico(ruta="mosaico.txt"):
    try:
        with open(ruta, "r") as f:
            lineas = f.read().splitlines()
        num_vecs = int(lineas[0])
        tamano = int(lineas[1])
        vectores = []
        for l in lineas[3:3+num_vecs]:
            vectores.append(np.array([int(x) for x in l.split()]))
        return vectores, tamano
    except FileNotFoundError:
        console.print("Error: No se encontró mosaico.txt.")
        return None, None

def imprimir_pixel_art(vector, titulo):
    console.print(f"\n{titulo}")
    caracteres = ["██" if x == 1 else "░░" for x in vector]
    for i in range(10):
        fila = "".join(caracteres[i*10:(i+1)*10])
        console.print(fila)
    console.print()

def main():
    vectores, tamano = cargar_mosaico()
    if vectores is None: return

    # Los primeros 10 son los patrones limpios del 0 al 9
    limpios = vectores[0:10]

    console.print("============================================================")
    console.print(" REPORTE INTERACTIVO: RED DE HAMMING (CLASIFICACIÓN)")
    console.print("============================================================")
    
    Prompt.ask("\n[FASE 1] Presiona ENTER para configurar las 10 neuronas con los patrones limpios...")
    
    # 100 píxeles, 10 patrones memorizados
    red = RedHamming(num_neuronas=tamano, num_patrones=len(limpios))
    red.entrenar(limpios)
    
    console.print("\nMatriz W (Capa de Similitud) configurada exitosamente.")
    console.print(f"Bias utilizado: {red.bias} | Epsilon (Inhibición): {red.epsilon:.4f}")
        
    Prompt.ask("\n[FASE 2] Presiona ENTER para clasificar vectores con ruido...")
    
    while True:
        console.print("-" * 60)
        seleccion = IntPrompt.ask("Ingresa el número del vector a clasificar (1-30) o '0' para salir")
        
        if seleccion == 0: break
            
        if seleccion < 1 or seleccion > len(vectores):
            console.print("Error: Índice fuera de rango.")
            continue
            
        idx = seleccion - 1
        vector_prueba = vectores[idx]
        numero_real = idx % 10
        
        imprimir_pixel_art(vector_prueba, f"VECTOR DE ENTRADA CON RUIDO (X_{seleccion})")
        
        # Predecir con la Red de Hamming
        estado, ganador, iteraciones, historial = red.predecir(vector_prueba)
        
        if estado == "CONVERGE":
            console.print(f"\n[ÉXITO] La capa MAXNET coronó a la neurona ganadora: {ganador}")
            console.print(f"La red clasificó el dibujo como el dígito: {ganador}")
            if ganador == numero_real:
                console.print("Diagnóstico: CLASIFICACIÓN CORRECTA.")
            else:
                console.print("Diagnóstico: CLASIFICACIÓN INCORRECTA (Demasiado ruido).")
        else:
            console.print("\n[FALLO] La red se anuló por empate. No pudo clasificar el dígito.")

        ver_tabla = Prompt.ask("¿Ver historial de la capa MAXNET?", choices=["s", "n"])
        if ver_tabla == "s":
            tabla = Table(title="Competencia de Neuronas (Solo se muestran las vivas)")
            tabla.add_column("Ronda")
            tabla.add_column("Valores de las 10 Neuronas")
            
            for it, valores in enumerate(historial):
                # Formateamos para que se vea legible
                vals_str = " ".join([f"{v:.2f}" if v > 0 else "  0  " for v in valores])
                tabla.add_row(str(it), vals_str)
            console.print(tabla)

if __name__ == "__main__":
    main()