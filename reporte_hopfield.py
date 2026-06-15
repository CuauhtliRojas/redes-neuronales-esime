import numpy as np
import sys
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from redes.hopfield import RedHopfield

console = Console(color_system=None)

np.set_printoptions(threshold=sys.maxsize, linewidth=250)


def cargar_mosaico(ruta="mosaico.txt"):
    try:
        with open(ruta, "r") as f:
            lineas = f.read().splitlines()

        num_vecs = int(lineas[0])
        tamano = int(lineas[1])
        vectores = []

        for l in lineas[3:3 + num_vecs]:
            vector = np.array([int(x) for x in l.split()], dtype=int)

            if len(vector) != tamano:
                raise ValueError("Un vector no coincide con el tamano declarado.")

            if not np.all(np.isin(vector, [-1, 1])):
                raise ValueError("Todos los vectores deben ser bipolares (-1 o 1).")

            vectores.append(vector)

        return vectores, tamano

    except FileNotFoundError:
        console.print("Error: No se encontro mosaico.txt. Ejecuta primero dibujador.py.")
        return None, None

    except ValueError as e:
        console.print(f"Error en mosaico.txt: {e}")
        return None, None


def imprimir_pixel_art(vector, titulo):
    console.print(f"\n{titulo}")

    lado = int(np.sqrt(len(vector)))

    if lado * lado != len(vector):
        console.print("Error: El vector no representa una matriz cuadrada.")
        return

    caracteres = ["██" if x == 1 else "░░" for x in vector]

    for i in range(lado):
        fila = "".join(caracteres[i * lado:(i + 1) * lado])
        console.print(fila)

    console.print()


def hamming(a, b):
    return int(np.sum(a != b))


def memoria_mas_cercana(x, memorias):
    distancias = [hamming(x, memoria) for memoria in memorias]
    idx = int(np.argmin(distancias))
    return idx, distancias[idx]


def entrenar_pseudoinversa(patrones, tamano):
    """
    Entrenamiento Hopfield por pseudoinversa.

    Se usa porque los digitos 10x10 estan altamente correlacionados.
    La red sigue siendo Hopfield; solo cambia la regla de calculo de W.
    """

    X = np.array(patrones, dtype=float)

    if X.shape[1] != tamano:
        raise ValueError("Los patrones no coinciden con el tamano de la red.")

    matriz_correlacion = X @ X.T
    W = X.T @ np.linalg.pinv(matriz_correlacion) @ X

    np.fill_diagonal(W, 0)

    return W


def predecir_hopfield(W, vector_inicial, max_iter=50):
    """
    Recuperacion Hopfield asincrona:
    - Estados bipolares {-1, 1}.
    - Si el campo local h_i es 0, conserva el valor anterior.
    """

    estado = vector_inicial.copy().astype(int)
    historial = []

    historial.append({
        "iteracion": 0,
        "estado": estado.copy(),
        "cambios": 0
    })

    for it in range(1, max_iter + 1):
        anterior = estado.copy()

        for i in range(len(estado)):
            h_i = W[i] @ estado

            if h_i > 0:
                estado[i] = 1
            elif h_i < 0:
                estado[i] = -1

        cambios = hamming(estado, anterior)

        historial.append({
            "iteracion": it,
            "estado": estado.copy(),
            "cambios": cambios
        })

        if cambios == 0:
            return "CONVERGE", estado.copy(), it, historial

    return "NO_CONVERGE", estado.copy(), max_iter, historial


def diagnosticar(distancia_objetivo, estado_red):
    if estado_red != "CONVERGE":
        return "NO ESTABLE"

    if distancia_objetivo == 0:
        return "EXITO"

    return "FALLO"


def imprimir_control_limpios(W, limpios):
    tabla = Table()
    tabla.add_column("Vector")
    tabla.add_column("Digito")
    tabla.add_column("Estado")
    tabla.add_column("Iter.")
    tabla.add_column("Error")
    tabla.add_column("Resultado")

    correctos = 0

    for i, patron in enumerate(limpios):
        estado_red, salida, iteraciones, _ = predecir_hopfield(W, patron)
        error = hamming(salida, patron)
        resultado = diagnosticar(error, estado_red)

        if error == 0 and estado_red == "CONVERGE":
            correctos += 1

        tabla.add_row(
            f"V_{i + 1}",
            str(i),
            estado_red,
            str(iteraciones),
            str(error),
            resultado
        )

    console.print(tabla)
    console.print(f"Resumen de estabilidad: {correctos}/10 patrones limpios estables.")

    if correctos == 10:
        console.print("Diagnostico global: La red almaceno correctamente las memorias limpias.")
    else:
        console.print("Diagnostico global: Algunas memorias limpias no son atractores estables.")


def imprimir_historial(historial, vector_objetivo, limpios):
    tabla = Table(title="Historial de recuperacion")
    tabla.add_column("Iter.")
    tabla.add_column("Cambios")
    tabla.add_column("Error vs objetivo")
    tabla.add_column("Memoria cercana")
    tabla.add_column("Dist.")
    tabla.add_column("Lectura")

    for registro in historial:
        it = registro["iteracion"]
        estado = registro["estado"]
        cambios = registro["cambios"]

        error_objetivo = hamming(estado, vector_objetivo)
        idx_cercano, dist_cercana = memoria_mas_cercana(estado, limpios)

        if it == 0:
            lectura = "Entrada inicial"
        elif cambios == 0 and error_objetivo == 0:
            lectura = "Estable correcto"
        elif cambios == 0 and error_objetivo > 0:
            lectura = "Estable incorrecto"
        elif error_objetivo == 0:
            lectura = "Ya reconstruido"
        else:
            lectura = "Ajustando"

        tabla.add_row(
            str(it),
            str(cambios),
            str(error_objetivo),
            f"V_{idx_cercano + 1} / digito {idx_cercano}",
            str(dist_cercana),
            lectura
        )

    console.print(tabla)


def main():
    vectores, tamano = cargar_mosaico()

    if vectores is None:
        return

    limpios = vectores[0:10]

    console.print("============================================================")
    console.print(" REPORTE INTERACTIVO: RED DE HOPFIELD (CORRECCION DE RUIDO)")
    console.print("============================================================")

    Prompt.ask("\n[FASE 1] Presiona ENTER para entrenar la red SOLO con los 10 patrones limpios...")

    red = RedHopfield(tamano=tamano, tipo_func='bipolar')
    red.entrenar(limpios)

    red.W = entrenar_pseudoinversa(limpios, tamano)

    console.print(f"\nMatriz W de pesos ({tamano}x{tamano}) calculada exitosamente.")

    imprimir_control_limpios(red.W, limpios)

    Prompt.ask("\n[FASE 2] Presiona ENTER para pasar a la Fase de Evaluacion con Ruido...")

    console.print("\nIndices de Vectores en tu archivo:")
    console.print(" - Del 1 al 10:  Patrones Limpios (Ideales)")
    console.print(" - Del 11 al 20: Patrones con 10% de Ruido")
    console.print(" - Del 21 al 30: Patrones con 35% de Ruido")

    while True:
        console.print("-" * 60)

        seleccion = IntPrompt.ask("Ingresa el numero del vector que deseas evaluar (1-30) o '0' para salir")

        if seleccion == 0:
            break

        if seleccion < 1 or seleccion > len(vectores):
            console.print("Error: Indice fuera de rango.")
            continue

        idx = seleccion - 1
        vector_prueba = vectores[idx]

        idx_limpio = idx % 10
        vector_objetivo = limpios[idx_limpio]

        imprimir_pixel_art(vector_prueba, f"VECTOR DE ENTRADA (V_{seleccion})")

        estado_red, estado_final, iteraciones, historial = predecir_hopfield(
            red.W,
            vector_prueba,
            max_iter=50
        )

        imprimir_pixel_art(estado_final, f"VECTOR DE SALIDA (Generado tras {iteraciones} iteraciones)")

        distancia_al_original = hamming(estado_final, vector_objetivo)
        idx_cercano, distancia_cercana = memoria_mas_cercana(estado_final, limpios)
        resultado = diagnosticar(distancia_al_original, estado_red)

        console.print(f"Estado de la red: {estado_red}")
        console.print(f"Patron esperado: V_{idx_limpio + 1} / digito {idx_limpio}")
        console.print(f"Distancia Hamming vs patron esperado: {distancia_al_original} bits de error")
        console.print(f"Memoria limpia mas cercana: V_{idx_cercano + 1} / digito {idx_cercano}")
        console.print(f"Distancia a la memoria mas cercana: {distancia_cercana}")
        console.print(f"Resultado: {resultado}")

        if resultado == "EXITO":
            console.print("Diagnostico: La red reconstruyo correctamente el patron limpio esperado.")
        elif estado_red == "CONVERGE":
            console.print("Diagnostico: La red converge, pero hacia una memoria incorrecta o deformada.")
        else:
            console.print("Diagnostico: La red no alcanzo un estado estable dentro del limite de iteraciones.")

        imprimir_historial(historial, vector_objetivo, limpios)


if __name__ == "__main__":
    main()