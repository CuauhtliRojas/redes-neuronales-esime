import numpy as np
import sys
from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from redes.BAM import RedBAM

console = Console(color_system=None)
np.set_printoptions(threshold=sys.maxsize, linewidth=250)


Y_ORTOGONALES = np.array([
    [-1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1],  # 0
    [-1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1],  # 1
    [-1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1],  # 2
    [-1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1],  # 3
    [-1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1],  # 4
    [-1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1],  # 5
    [-1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1],  # 6
    [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1],  # 7
    [-1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1],  # 8
    [-1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1]   # 9
], dtype=int)


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
        console.print("Error: No se encontro mosaico.txt.")
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


def signo_bipolar(v):
    return np.where(v >= 0, 1, -1).astype(int)


def predecir_ida_bam(red, x):
    return signo_bipolar(x @ red.W)


def predecir_vuelta_bam(red, y):
    return signo_bipolar(y @ red.W.T)


def validar_codigos_ortogonales():
    productos = Y_ORTOGONALES @ Y_ORTOGONALES.T
    diagonal_correcta = np.all(np.diag(productos) == Y_ORTOGONALES.shape[1])

    fuera_diagonal = productos.copy()
    np.fill_diagonal(fuera_diagonal, 0)
    ortogonales = np.all(fuera_diagonal == 0)

    return diagonal_correcta and ortogonales


def decodificar_y(vector):
    distancias = [hamming(vector, y) for y in Y_ORTOGONALES]
    min_dist = min(distancias)
    numero_estimado = int(distancias.index(min_dist))
    return numero_estimado, int(min_dist)


def canonizar_y(vector):
    numero, error_bits = decodificar_y(vector)
    return Y_ORTOGONALES[numero].copy(), numero, error_bits


def memoria_x_mas_cercana(x, limpios_x):
    distancias = [hamming(x, patron) for patron in limpios_x]
    idx = int(np.argmin(distancias))
    return idx, int(distancias[idx])


def clasificar_desde_x(red, x):
    y_crudo = predecir_ida_bam(red, x)
    y_canonico, numero, error_y = canonizar_y(y_crudo)

    return {
        "y_crudo": y_crudo,
        "y_canonico": y_canonico,
        "numero": numero,
        "error_y": error_y
    }


def reconstruir_desde_y(red, y):
    return predecir_vuelta_bam(red, y)


def resonancia_controlada(red, y_referencia, numero_referencia, limpios_x, max_iter=10):
    """
    Resonancia BAM controlada.

    Se usa Y canonico como ancla para evitar que un error pequeno en Y
    arrastre la red hacia otro atractor durante el ciclo X <-> Y.

    Esto mantiene la asociacion BAM:
    Y canonico -> X reconstruido -> validacion X -> Y
    """

    historial = []
    y_actual = y_referencia.copy()
    x_anterior = None
    x_final = None

    for it in range(1, max_iter + 1):
        x_actual = reconstruir_desde_y(red, y_actual)
        y_crudo = predecir_ida_bam(red, x_actual)
        y_validado, numero_validado, error_y = canonizar_y(y_crudo)

        dist_x_referencia = hamming(x_actual, limpios_x[numero_referencia])
        idx_x_cercano, dist_x_cercano = memoria_x_mas_cercana(x_actual, limpios_x)

        if numero_validado != numero_referencia:
            lectura = f"Se desviaria a {numero_validado}; se detiene"
            historial.append({
                "iteracion": it,
                "x": x_actual.copy(),
                "numero_validado": numero_validado,
                "error_y": error_y,
                "dist_x_referencia": dist_x_referencia,
                "idx_x_cercano": idx_x_cercano,
                "dist_x_cercano": dist_x_cercano,
                "lectura": lectura
            })

            return "DESVIO_CONTROLADO", it, x_actual.copy(), historial

        if x_anterior is not None and np.array_equal(x_actual, x_anterior):
            lectura = "Estable"
            historial.append({
                "iteracion": it,
                "x": x_actual.copy(),
                "numero_validado": numero_validado,
                "error_y": error_y,
                "dist_x_referencia": dist_x_referencia,
                "idx_x_cercano": idx_x_cercano,
                "dist_x_cercano": dist_x_cercano,
                "lectura": lectura
            })

            return "RESONANCIA", it, x_actual.copy(), historial

        lectura = "Asociacion conservada"
        historial.append({
            "iteracion": it,
            "x": x_actual.copy(),
            "numero_validado": numero_validado,
            "error_y": error_y,
            "dist_x_referencia": dist_x_referencia,
            "idx_x_cercano": idx_x_cercano,
            "dist_x_cercano": dist_x_cercano,
            "lectura": lectura
        })

        x_anterior = x_actual.copy()
        x_final = x_actual.copy()

    return "LIMITE_ITERACIONES", max_iter, x_final.copy(), historial


def imprimir_control_asociaciones(red, limpios_x):
    tabla = Table(title="Control de asociaciones limpias")
    tabla.add_column("Digito")
    tabla.add_column("X -> Y")
    tabla.add_column("Err. Y")
    tabla.add_column("Y -> X")
    tabla.add_column("X cercana")
    tabla.add_column("Estado")

    correctos_xy = 0
    correctos_yx = 0

    for i, x_limpio in enumerate(limpios_x):
        resultado_xy = clasificar_desde_x(red, x_limpio)
        num_xy = resultado_xy["numero"]
        err_y = resultado_xy["error_y"]

        y_limpio = Y_ORTOGONALES[i]
        x_recon = reconstruir_desde_y(red, y_limpio)

        dist_yx = hamming(x_recon, x_limpio)
        idx_cercano, dist_cercana = memoria_x_mas_cercana(x_recon, limpios_x)

        ok_xy = num_xy == i
        ok_yx = idx_cercano == i

        if ok_xy:
            correctos_xy += 1

        if ok_yx:
            correctos_yx += 1

        if ok_xy and ok_yx:
            estado = "OK"
        elif ok_xy:
            estado = "Parcial"
        else:
            estado = "Fallo"

        tabla.add_row(
            str(i),
            f"{num_xy}",
            str(err_y),
            f"{dist_yx} bits",
            f"{idx_cercano} ({dist_cercana})",
            estado
        )

    console.print(tabla)
    console.print(f"Resumen X -> Y: {correctos_xy}/10 clasificaciones limpias correctas.")
    console.print(f"Resumen Y -> X: {correctos_yx}/10 reconstrucciones cercanas correctas.")


def imprimir_historial_controlado(historial):
    tabla = Table(title="Historial de resonancia controlada")
    tabla.add_column("Iter.")
    tabla.add_column("Y valida")
    tabla.add_column("Err. Y")
    tabla.add_column("Dist. X ref.")
    tabla.add_column("X cercana")
    tabla.add_column("Lectura")

    for registro in historial:
        tabla.add_row(
            str(registro["iteracion"]),
            str(registro["numero_validado"]),
            str(registro["error_y"]),
            str(registro["dist_x_referencia"]),
            f"{registro['idx_x_cercano']} ({registro['dist_x_cercano']})",
            registro["lectura"]
        )

    console.print(tabla)


def diagnostico_bam(numero_esperado, numero_obtenido, dist_x, estado_resonancia):
    if numero_obtenido != numero_esperado:
        return "FALLO_ASOCIACION"

    if estado_resonancia == "DESVIO_CONTROLADO":
        return "ASOCIACION_CORRECTA_RESONANCIA_INESTABLE"

    if dist_x == 0:
        return "EXITO_TOTAL"

    return "ASOCIACION_CORRECTA_RECONSTRUCCION_APROXIMADA"


def main():
    vectores, tamano_x = cargar_mosaico()

    if vectores is None:
        return

    limpios_x = vectores[0:10]
    tamano_y = 16

    console.print("============================================================")
    console.print(" REPORTE INTERACTIVO: RED BAM CON CODIFICACION ORTOGONAL")
    console.print(" Imagen 100 px (X) <-> Codigo ortogonal 16 bits (Y)")
    console.print("============================================================")

    if not validar_codigos_ortogonales():
        console.print("Error: Los codigos Y no son ortogonales. Revisa Y_ORTOGONALES.")
        return

    Prompt.ask("\n[FASE 1] Presiona ENTER para entrenar la red BAM con los 10 patrones limpios...")

    red = RedBAM(di=tamano_x, do=tamano_y)
    red.entrenar(limpios_x, Y_ORTOGONALES)

    console.print(f"\nMatriz W bidireccional ({tamano_x}x{tamano_y}) calculada exitosamente.")
    imprimir_control_asociaciones(red, limpios_x)

    while True:
        console.print("\n" + "=" * 60)
        console.print("[FASE 2] SELECCION DE PRUEBA")
        console.print("1. Probar propagacion X -> Y (Ingresar dibujo con/sin ruido)")
        console.print("2. Probar propagacion Y -> X (Ingresar digito 0-9)")
        console.print("0. Salir")

        opcion = Prompt.ask("Selecciona una opcion", choices=["1", "2", "0"])

        if opcion == "0":
            break

        elif opcion == "1":
            seleccion = IntPrompt.ask("\nIngresa el numero del vector X (1-30)")

            if seleccion < 1 or seleccion > len(vectores):
                console.print("Indice fuera de rango.")
                continue

            idx = seleccion - 1
            idx_limpio = idx % 10

            vec_prueba = vectores[idx]
            vec_objetivo = limpios_x[idx_limpio]

            imprimir_pixel_art(vec_prueba, f"INICIO: DIBUJO DE ENTRADA (X_{seleccion})")

            resultado_xy = clasificar_desde_x(red, vec_prueba)
            numero_obtenido = resultado_xy["numero"]
            error_y = resultado_xy["error_y"]
            y_crudo = resultado_xy["y_crudo"]
            y_canonico = resultado_xy["y_canonico"]

            estado, iteraciones, x_final, historial = resonancia_controlada(
                red,
                y_canonico,
                numero_obtenido,
                limpios_x
            )

            dist_entrada = hamming(vec_prueba, vec_objetivo)
            dist_final_objetivo = hamming(x_final, vec_objetivo)
            idx_cercano, dist_cercana = memoria_x_mas_cercana(x_final, limpios_x)

            diagnostico = diagnostico_bam(
                idx_limpio,
                numero_obtenido,
                dist_final_objetivo,
                estado
            )

            console.print("\n--- RESULTADO X -> Y ---")
            console.print(f"Patron esperado: digito {idx_limpio}")
            console.print(f"Distancia inicial vs limpio esperado: {dist_entrada} bits")
            console.print(f"Y crudo calculado: {y_crudo}")
            console.print(f"Y canonico usado:  {y_canonico}")
            console.print(f"Digito reconocido: {numero_obtenido}")
            console.print(f"Error del Y crudo vs codigo canonico: {error_y} bits")

            console.print("\n--- RESULTADO Y -> X ---")
            console.print(f"Estado de resonancia: {estado} ({iteraciones} iteraciones)")
            console.print(f"Distancia final vs limpio esperado: {dist_final_objetivo} bits")
            console.print(f"Imagen limpia mas cercana: digito {idx_cercano} con distancia {dist_cercana}")
            console.print(f"Diagnostico: {diagnostico}")

            imprimir_historial_controlado(historial)
            imprimir_pixel_art(x_final, "DIBUJO FINAL RECONSTRUIDO POR LA RED (X Final)")

        elif opcion == "2":
            entrada = Prompt.ask("\nIngresa el digito (0-9) a probar")

            try:
                num = int(entrada)
            except ValueError:
                console.print("Entrada invalida.")
                continue

            if num < 0 or num > 9:
                console.print("Error: El digito debe ser del 0 al 9.")
                continue

            vec_y = Y_ORTOGONALES[num].copy()

            console.print(f"\nINICIO: VECTOR Y canonico del digito {num}")
            console.print(vec_y)

            estado, iteraciones, x_final, historial = resonancia_controlada(
                red,
                vec_y,
                num,
                limpios_x
            )

            idx_cercano, dist_cercana = memoria_x_mas_cercana(x_final, limpios_x)
            dist_objetivo = hamming(x_final, limpios_x[num])

            if estado == "DESVIO_CONTROLADO":
                diagnostico = "ASOCIACION_GENERADA_PERO_RESONANCIA_LIBRE_SE_DESVIARIA"
            elif dist_objetivo == 0:
                diagnostico = "EXITO_TOTAL"
            else:
                diagnostico = "RECONSTRUCCION_APROXIMADA"

            console.print("\n--- RESULTADO Y -> X ---")
            console.print(f"Digito solicitado: {num}")
            console.print(f"Estado de resonancia: {estado} ({iteraciones} iteraciones)")
            console.print(f"Distancia vs imagen limpia solicitada: {dist_objetivo} bits")
            console.print(f"Imagen limpia mas cercana: digito {idx_cercano} con distancia {dist_cercana}")
            console.print(f"Diagnostico: {diagnostico}")

            imprimir_historial_controlado(historial)
            imprimir_pixel_art(x_final, "DIBUJO FINAL RECONSTRUIDO POR LA RED (X Final)")


if __name__ == "__main__":
    main()