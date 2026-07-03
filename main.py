import numpy as np
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.table import Table
from rich.panel import Panel
from pathlib import Path

from redes.hopfield import RedHopfield
from redes.BAM import RedBAM  
from redes.hamming import RedHamming
from redes.perceptron import PerceptronSimple

console = Console()

def capturar_vector(mensaje, tamano, permite_salir=False):
    tamanos_validos = tamano if isinstance(tamano, list) else [tamano]
    
    while True:
        entrada = Prompt.ask(mensaje)
        
        if permite_salir and entrada.lower() in ['salir']:
            return None
            
        try:
            vec = np.array([float(x) for x in entrada.split()])
            
            if len(vec) not in tamanos_validos:
                if len(tamanos_validos) == 1:
                    console.print(f"Error: El vector debe tener {tamanos_validos[0]} elementos.")
                else:
                    console.print(f"Error: El vector debe tener {tamanos_validos[0]} o {tamanos_validos[1]} elementos.")
                continue
            return vec
        except ValueError:
            console.print("Error: Ingresa solo números separados por espacio.")

def cargar_entrenamiento_desde_archivo(ruta_archivo):
    ruta = Path(ruta_archivo)
    if not ruta.exists(): raise FileNotFoundError(f"No existe el archivo: {ruta_archivo}")
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    lineas = [linea.strip() for linea in lineas if linea.strip()]
    if len(lineas) < 3: raise ValueError("El archivo debe tener al menos 3 líneas.")
    
    num_vectores = int(lineas[0])
    tamano_vector = int(lineas[1])
    opcion_func = lineas[2]
    tipo_func = "bipolar" if opcion_func == "1" else "binario"
    
    patrones = []
    for i in range(num_vectores):
        vector = np.array([int(x) for x in lineas[3 + i].split()])
        patrones.append(vector)
    return num_vectores, tamano_vector, tipo_func, patrones

def ejecutar_hopfield():
    console.print(Panel.fit("Módulo: Red de Hopfield"))
    modo_carga = Prompt.ask("¿Cómo quieres cargar los patrones? 1 Archivo o 2 Manual", choices=["1", "2"])

    if modo_carga == "1":
        archivo_entrada = "entrada_test.txt"
        try:
            num_vectores, tamano_vector, tipo_func, patrones = cargar_entrenamiento_desde_archivo(archivo_entrada)
            console.print(f"\nDatos cargados de {archivo_entrada}")
            console.print(f"Vectores: {num_vectores}")
            console.print(f"Tamaño de vector: {tamano_vector}")
            console.print(f"Función: {tipo_func}")
        except Exception as e:
            console.print(f"Error al cargar archivo: {e}")
            return
    else:
        num_vectores = IntPrompt.ask("¿Cuántos vectores?")
        tamano_vector = IntPrompt.ask("¿Cuál es el tamaño de cada vector?")
        opcion_func = Prompt.ask("¿Función de transferencia? 1 Bipolar o 2 Binaria", choices=["1", "2"])
        tipo_func = "bipolar" if opcion_func == "1" else "binario"
        patrones = []
        console.print("\nIngresa los vectores:")
        for i in range(num_vectores):
            vec = capturar_vector(f"Vector {i + 1}", tamano_vector)
            patrones.append(vec)

    red = RedHopfield(tamano=tamano_vector, tipo_func=tipo_func)
    desglose = red.entrenar(patrones)

    console.print("\nRed Entrenada Exitosamente.")
    console.print("Cálculo de pesos:")
    for formula in desglose:
        console.print(formula)
        
    console.print("\nMatriz de Pesos (W):")
    console.print(red.W)

    console.print("\n--- FASE DE PRUEBA ---")
    console.print("La red está lista para probar vectores.")
    
    while True:
        console.print("\n[Escribe 'salir' para terminar la ejecución]")
        Xp = capturar_vector("Ingresa el vector de PRUEBA a evaluar", tamano_vector, permite_salir=True)
        
        if Xp is None:
            console.print("\nCerrando..")
            break
        
        estado_red, estado_final, iteraciones, dist_final, historial = red.predecir(Xp)

        tabla = Table(title="Historial de Iteraciones")
        tabla.add_column("Iteración", justify="center")
        tabla.add_column("Estado del Vector")
        tabla.add_column("Distancia a Xp", justify="right")

        for it, estado, distancia in historial:
            tabla.add_row(str(it), str(estado), str(distancia))
        
        console.print(tabla)

        if estado_red == "converge":
            patron_asociado = "Ninguno (Convergió a un patrón espurio/desconocido)"
            for idx, p in enumerate(patrones):
                if np.array_equal(estado_final, p):
                    patron_asociado = f"Patrón {idx + 1}"
                    break

            distancias_patrones = red.encontrar_patron_mas_cercano(estado_final, patrones)
            patron_mas_cercano, distancia_minima = distancias_patrones[0]
                    
            mensaje = (
                f"Convergencia alcanzada (Iteración {iteraciones})\n"
                f"Vector Final: {estado_final}\n"
                f"Bits modificados desde Xp: {dist_final}\n"
                f"Resultado exacto: {patron_asociado}\n"
                f"Patrón más cercano: Patrón {patron_mas_cercano} (Distancia: {distancia_minima})"
            )
            console.print(Panel(mensaje, expand=False))
            
        elif estado_red == "bucle":
            mensaje = (
                f"El vector entró en un BUCLE infinito.\n"
                f"Se detectó la oscilación en la iteración {iteraciones}.\n"
                f"La red está rebotando entre estados y nunca convergerá.\n"
                f"Último estado antes de detectar el bucle: {estado_final}"
            )
            console.print(Panel(mensaje, expand=False))
            
        else:
            mensaje = (
                f"La red no convergió.\n"
                f"Se alcanzó el límite máximo de {iteraciones} iteraciones.\n"
                f"Último estado: {estado_final}"
            )
            console.print(Panel(mensaje, expand=False))

def ejecutar_bam():
    console.print(Panel.fit("Módulo: Memoria Asociativa Bidireccional (BAM)"))
    
    pp = IntPrompt.ask("Ingresa la cantidad de pares de patrones (X, Y)")
    di = IntPrompt.ask("Ingresa las dimensiones de los vectores de ENTRADA (X)")
    do = IntPrompt.ask("Ingresa las dimensiones de los vectores de SALIDA (Y)")

    patrones_input = []
    patrones_output = []

    console.print("\nCaptura de patrones:")
    for i in range(pp):
        console.print(f"\n--- Par {i+1} ---")
        vec_in = capturar_vector(f"Vector de ENTRADA X{i+1}", di)
        vec_out = capturar_vector(f"Vector de SALIDA Y{i+1}", do)
        patrones_input.append(vec_in)
        patrones_output.append(vec_out)

    red = RedBAM(di=di, do=do)
    red.entrenar(patrones_input, patrones_output)

    console.print("\nRed BAM Entrenada Exitosamente.")
    console.print("\nMatriz de Pesos (W):\n", red.W)

    console.print("\n--- FASE DE PRUEBA BAM ---")
    console.print("El sistema detectará automáticamente si ingresas un vector X (Ida) o Y (Vuelta) basado en su tamaño.")
    
    # Tamaños válidos para autodetectar
    tamanos_permitidos = list(set([di, do])) 

    while True:
        console.print("\n[Escribe 'salir' para terminar]")
        vec_prueba = capturar_vector(f"Ingresa un vector de prueba (Tamaño {di} para X, o {do} para Y)", tamanos_permitidos, permite_salir=True)
        
        if vec_prueba is None: 
            break
            
        tamano_ingresado = len(vec_prueba)
        es_ida = (tamano_ingresado == di)
        es_vuelta = (tamano_ingresado == do)

        # Resolución de ambigüedad si X y Y tienen el mismo tamaño
        if es_ida and es_vuelta:
            direccion = Prompt.ask("El tamaño coincide con entrada y salida. ¿Es [1] Entrada (X) o [2] Salida (Y)?", choices=["1", "2"])
            if direccion == "2":
                es_ida = False

        if es_ida:
            console.print("\nDetectado: Operación X -> Y (Ida)")
            resultado = red.predecir_ida(vec_prueba)
            console.print(f"Resultado (Y): {resultado}")
            
            asociado = False
            for i, p_out in enumerate(patrones_output):
                if np.array_equal(resultado, p_out):
                    console.print(f"Asocia correctamente con la salida Y{i+1}")
                    asociado = True
            if not asociado:
                console.print("No asocia con ningún vector de salida memorizado.")
                
        else:
            console.print("\nDetectado: Operación Y -> X (Vuelta)")
            resultado = red.predecir_vuelta(vec_prueba)
            console.print(f"Resultado (X): {resultado}")
            
            asociado = False
            for i, p_in in enumerate(patrones_input):
                if np.array_equal(resultado, p_in):
                    console.print(f"Asocia correctamente con la entrada X{i+1}")
                    asociado = True
            if not asociado:
                console.print("No asocia con ningún vector de entrada memorizado.")

def ejecutar_hamming():
    console.print(Panel.fit("Módulo: Red de Hamming (Clasificador MAXNET)"))
    
    n_neuronas = IntPrompt.ask("¿Número de neuronas (Tamaño del vector N)?")
    n_patrones = IntPrompt.ask("¿Número de patrones a memorizar (M)?")
    
    patrones = []
    console.print("\nIngresa los vectores de entrenamiento (bipolares 1 o -1):")
    for i in range(n_patrones):
        vec = capturar_vector(f"Patrón de memoria e{i + 1}", n_neuronas)
        patrones.append(vec)
        
    red = RedHamming(num_neuronas=n_neuronas, num_patrones=n_patrones)
    red.entrenar(patrones)
    console.print("\nRed Hamming Entrenada. Matriz de Similitud (W):")
    console.print(red.W)
    
    console.print("\n--- FASE DE PRUEBA ---")
    while True:
        console.print("\n[Escribe 'salir' para terminar]")
        p = capturar_vector("Ingresa vector de prueba", n_neuronas, permite_salir=True)
        if p is None: break
        
        estado, ganador, iteraciones, historial = red.predecir(p)
        
        tabla = Table(title="Competencia MAXNET")
        tabla.add_column("Iteración", justify="center")
        tabla.add_column("Valores de las Neuronas")
        
        for it, valores in enumerate(historial):
            tabla.add_row(str(it), str(np.round(valores, 4)))
            
        console.print(tabla)
        if estado == "CONVERGE":
            console.print(f"[EXITO] La red asocia el vector con la Memoria e{ganador + 1}")
        else:
            console.print("[FALLO] Las neuronas empataron y se anularon mutuamente (Resultado Nulo).")

def ejecutar_perceptron():
    console.print(Panel.fit("Módulo: Perceptrón Simple"))
    
    num_entradas = IntPrompt.ask("¿Número de entradas (tamaño del vector X)?")
    num_patrones = IntPrompt.ask("¿Cuántos pares de entrenamiento (X -> Y) ingresarás?")
    tasa_lr = FloatPrompt.ask("Tasa de aprendizaje (Alpha)", default=1.0)
    
    # --- NUEVA SECCIÓN DE INICIALIZACIÓN ---
    console.print("\n[Inicialización]")
    tipo_init = Prompt.ask("¿Cómo deseas iniciar los Pesos (W)?", choices=["1", "2", "3"], default="1")
    # 1=Ceros, 2=Random, 3=Manual
    
    pesos_init = None
    if tipo_init == "3":
        pesos_init = capturar_vector("Ingresa los pesos iniciales W separados por espacio", num_entradas)
    elif tipo_init == "2":
        pesos_init = np.random.uniform(-1, 1, num_entradas)
        console.print(f"Pesos random generados: {np.round(pesos_init, 2)}")
        
    bias_init = FloatPrompt.ask("Ingresa el Bias inicial (b) o Theta", default=0.0)
    
    X_train, Y_train = [], []
    console.print("\nCaptura de pares de entrenamiento:")
    for i in range(num_patrones):
        console.print(f"\n--- Patrón {i+1} ---")
        x = capturar_vector("Vector de entrada X", num_entradas)
        y = FloatPrompt.ask("Salida deseada Y (-1 o 1)")
        X_train.append(x)
        Y_train.append(y)
        
    # Inicializamos la red con los datos recabados
    red = PerceptronSimple(
        num_entradas=num_entradas, 
        tasa_aprendizaje=tasa_lr, 
        pesos_iniciales=pesos_init, 
        bias_inicial=bias_init
    )
    
    historial = red.entrenar(X_train, Y_train)
    
    tabla = Table(title="Historial de Entrenamiento (Regla Delta)")
    tabla.add_column("Época", justify="center")
    tabla.add_column("Pesos (W)")
    tabla.add_column("Bias (b)")
    tabla.add_column("Precisión (%)", justify="center", style="bold green")
    
    # Calculamos el porcentaje fila por fila
    for ep, w, b, err in historial:
        aciertos = num_patrones - err
        porcentaje = (aciertos / num_patrones) * 100
        
        tabla.add_row(
            str(ep), 
            str(np.round(w, 4)), 
            str(round(b, 4)), 
            f"{porcentaje:.1f}%"  # Formatea a 1 decimal y le pone el signo %
        )
        
    console.print(tabla)
    
    console.print("\n--- FASE DE PRUEBA ---")
    while True:
        p = capturar_vector("\nIngresa vector de prueba (o 'salir')", num_entradas, permite_salir=True)
        if p is None: break
        
        # Predicción interna
        pred = red.predecir(p)
        
        # Desglose Matemático para la consola
        w_str = f"[{', '.join(str(round(w, 4)) for w in red.W)}]"
        p_str = f"[{', '.join(str(round(x, 4)) for x in p)}]"
        
        # Armamos la cadena de multiplicaciones "(W1*X1) + (W2*X2)..."
        operaciones = " + ".join([f"({round(w,4)} * {round(x,4)})" for w, x in zip(red.W, p)])
        suma_productos = np.dot(red.W, p)
        net = suma_productos + red.b
        
        console.print(Panel((
            f"[bold cyan]1. Datos actuales:[/bold cyan]\n"
            f"   Pesos (W) = {w_str}\n"
            f"   Entrada (X) = {p_str}\n"
            f"   Bias (b) = {round(red.b, 4)}\n\n"
            f"[bold cyan]2. Producto Punto (W · X):[/bold cyan]\n"
            f"   {operaciones} = {round(suma_productos, 4)}\n\n"
            f"[bold cyan]3. Suma del Bias (Net):[/bold cyan]\n"
            f"   {round(suma_productos, 4)} + ({round(red.b, 4)}) = [bold yellow]{round(net, 4)}[/bold yellow]\n\n"
            f"[bold cyan]4. Función de Activación:[/bold cyan]\n"
            f"   Como {round(net, 4)} {'[bold green]>= 0[/bold green]' if net >= 0 else '[bold red]< 0[/bold red]'}, "
            f"la salida es: [bold magenta]{pred}[/bold magenta]"
        ), title="Desglose Matemático de la Predicción", expand=False))
def main():
    while True:
        console.print(Panel.fit("Simulador de Redes Neuronales\nElige el modelo que deseas ejecutar."))
        console.print("1. Red de Hopfield")
        console.print("2. Memoria Asociativa Bidireccional (BAM)")
        console.print("3. Red de Hamming")
        console.print("4. Perceptrón Simple")
        console.print("0. Salir")
        
        opcion = Prompt.ask("Selecciona", choices=["1", "2", "3", "4", "0"])
        
        if opcion == "1": ejecutar_hopfield()
        elif opcion == "2": ejecutar_bam()
        elif opcion == "3": ejecutar_hamming()
        elif opcion == "4": ejecutar_perceptron()
        elif opcion == "0": break

if __name__ == "__main__":
    main()