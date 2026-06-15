import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import simpledialog
from rich.console import Console
from rich.prompt import Prompt, IntPrompt

console = Console(color_system=None)

# Plantillas base
NUMEROS_ASCII = [
    "..........\n..XXXXXX..\n.XX....XX.\n.XX....XX.\n.XX....XX.\n.XX....XX.\n.XX....XX.\n.XX....XX.\n..XXXXXX..\n..........", # 0
    "..........\n....XX....\n...XXX....\n..XXXX....\n....XX....\n....XX....\n....XX....\n....XX....\n..XXXXXX..\n..........", # 1
    "..........\n..XXXXXX..\n.XX....XX.\n.......XX.\n......XX..\n.....XX...\n....XX....\n...XX.....\n.XXXXXXXX.\n..........", # 2
    "..........\n..XXXXXX..\n.XX....XX.\n.......XX.\n...XXXXX..\n.......XX.\n.......XX.\n.XX....XX.\n..XXXXXX..\n..........", # 3
    "..........\n......XX..\n.....XXX..\n....XXXX..\n...XX.XX..\n..XX..XX..\n.XXXXXXXX.\n......XX..\n......XX..\n..........", # 4
    "..........\n.XXXXXXXX.\n.XX.......\n.XXXXXXX..\n.......XX.\n.......XX.\n.......XX.\n.XX....XX.\n..XXXXXX..\n..........", # 5
    "..........\n...XXXXX..\n..XX......\n.XX.......\n.XXXXXXX..\n.XX....XX.\n.XX....XX.\n.XX....XX.\n..XXXXXX..\n..........", # 6
    "..........\n.XXXXXXXX.\n.......XX.\n......XX..\n.....XX...\n....XX....\n...XX.....\n..XX......\n..XX......\n..........", # 7
    "..........\n..XXXXXX..\n.XX....XX.\n.XX....XX.\n..XXXXXX..\n.XX....XX.\n.XX....XX.\n.XX....XX.\n..XXXXXX..\n..........", # 8
    "..........\n..XXXXXX..\n.XX....XX.\n.XX....XX.\n..XXXXXXX.\n.......XX.\n......XX..\n.....XX...\n..XXXX....\n.........."  # 9
]

def texto_a_vector(texto):
    plano = texto.replace('\n', '')
    return np.array([1 if char == 'X' else -1 for char in plano])

def aplicar_ruido(vector, porcentaje):
    vec_ruido = vector.copy()
    num_cambios = int(len(vector) * (porcentaje / 100.0))
    indices = np.random.choice(len(vector), num_cambios, replace=False)
    for idx in indices:
        vec_ruido[idx] *= -1 
    return vec_ruido

def vector_a_imagen(vector, tamano_celda=20):
    img = Image.new('RGB', (10 * tamano_celda, 10 * tamano_celda), color='white')
    draw = ImageDraw.Draw(img)
    
    for i, val in enumerate(vector):
        fila = i // 10
        col = i % 10
        color = 'black' if val == 1 else 'white'
        
        x0 = col * tamano_celda
        y0 = fila * tamano_celda
        x1 = x0 + tamano_celda
        y1 = y0 + tamano_celda
        
        draw.rectangle([x0, y0, x1, y1], fill=color, outline='gray')
    return img

def exportar_txt(nombre_archivo, vectores):
    """Genera el archivo TXT formateado para que Hopfield/BAM lo lea directo."""
    with open(nombre_archivo, "w") as f:
        f.write(f"{len(vectores)}\n100\n1\n") # Cantidad, Tamano(100), Funcion(1=Bipolar)
        for vec in vectores:
            f.write(" ".join(map(str, vec)) + "\n")

def crear_mosaico_reporte():
    console.print("\n--- GENERADOR DE MOSAICO DINAMICO ---")
    usar_ruido = Prompt.ask("¿Quieres aplicar filas de ruido extra?", choices=["s", "n"])
    
    filas_ruido = 0
    porcentajes = []
    if usar_ruido == "s":
        filas_ruido = IntPrompt.ask("¿Cuantas filas de ruido quieres generar?")
        for i in range(filas_ruido):
            p = IntPrompt.ask(f"Porcentaje de ruido para la fila {i+1} (ej. 10, 35)")
            porcentajes.append(p)

    # Procesamiento de vectores
    limpios = [texto_a_vector(num) for num in NUMEROS_ASCII]
    vectores_totales = list(limpios)
    
    filas_generadas = [("Patrones Limpios (0% Ruido)", limpios)]
    for p in porcentajes:
        ruidosos = [aplicar_ruido(v, p) for v in limpios]
        filas_generadas.append((f"Patrones con {p}% de Ruido", ruidosos))
        vectores_totales.extend(ruidosos)

    # Dibujo del lienzo
    tamano_celda = 20
    ancho_img = 10 * tamano_celda
    alto_img = 10 * tamano_celda
    alto_fila = alto_img + 30 # Espacio extra para la etiqueta
    
    mosaico = Image.new('RGB', (10 * ancho_img, len(filas_generadas) * alto_fila), color='white')
    draw = ImageDraw.Draw(mosaico)

    for fila_idx, (titulo, lista_vec) in enumerate(filas_generadas):
        y_offset = fila_idx * alto_fila
        # Etiqueta de la fila
        draw.text((10, y_offset + 5), titulo, fill="black")
        
        for col_idx, vec in enumerate(lista_vec):
            img_v = vector_a_imagen(vec, tamano_celda)
            mosaico.paste(img_v, (col_idx * ancho_img, y_offset + 30))

    mosaico.save("reporte_mosaico.png")
    exportar_txt("mosaico.txt", vectores_totales)
    
    console.print("\n[EXITO] Imagen guardada como: reporte_mosaico.png")
    console.print(f"[EXITO] Archivo de datos guardado: mosaico.txt (Contiene {len(vectores_totales)} vectores)")

def traducir_vector_manual():
    console.print("\n--- TRADUCTOR: VECTOR -> IMAGEN ---")
    entrada = Prompt.ask("Ingresa los 100 numeros del vector (separados por espacio)")
    try:
        vec = np.array([int(x) for x in entrada.split()])
        if len(vec) != 100:
            console.print("Error: El vector debe tener 100 elementos.")
            return
            
        img = vector_a_imagen(vec)
        nombre = Prompt.ask("Nombre para guardar (ej. prueba.png)")
        if not nombre.endswith(".png"): nombre += ".png"
        img.save(nombre)
        console.print(f"[EXITO] Guardado exitosamente como {nombre}")
    except ValueError:
        console.print("Error: Ingresa solo numeros.")

def abrir_interfaz_buscaminas():
    console.print("\n--- MODO DIBUJO MULTIPLE ---")
    num_dibujos = IntPrompt.ask("¿Cuantos dibujos quieres hacer para tu set de entrenamiento?")
    
    vectores_creados = []

    for i in range(num_dibujos):
        console.print(f"Abriendo lienzo {i+1} de {num_dibujos}...")
        
        ventana = tk.Tk()
        ventana.title(f"Dibujando patron {i+1}/{num_dibujos} - ESIME")
        ventana.resizable(False, False)
        
        estado_vector = [-1] * 100
        botones = []

        def toggle_pixel(idx, v=estado_vector, b=botones):
            v[idx] *= -1
            color = "black" if v[idx] == 1 else "white"
            b[idx].config(bg=color)

        frame_cuadricula = tk.Frame(ventana, bg="gray")
        frame_cuadricula.pack(padx=20, pady=20)

        for j in range(100):
            btn = tk.Button(frame_cuadricula, width=4, height=2, bg="white", relief="raised",
                            command=lambda idx=j: toggle_pixel(idx))
            btn.grid(row=j//10, column=j%10, padx=1, pady=1)
            botones.append(btn)

        def guardar_y_cerrar():
            ventana.quit()

        btn_guardar = tk.Button(ventana, text="Guardar y Continuar", font=("Arial", 12, "bold"), command=guardar_y_cerrar)
        btn_guardar.pack(pady=10)
        
        ventana.mainloop()
        
        # Una vez que se presiona continuar, guardamos y destruimos la ventana
        vectores_creados.append(np.array(estado_vector))
        ventana.destroy()

    # Creacion del mosaico final de los dibujos
    tamano_celda = 20
    ancho_img = 10 * tamano_celda
    alto_img = 10 * tamano_celda
    alto_fila = alto_img + 30
    
    ancho_mosaico = num_dibujos * ancho_img
    mosaico = Image.new('RGB', (ancho_mosaico, alto_fila), color='white')
    draw = ImageDraw.Draw(mosaico)
    
    draw.text((10, 5), f"Vectores Dibujados Manualmente ({num_dibujos})", fill="black")
    
    for col_idx, vec in enumerate(vectores_creados):
        img_v = vector_a_imagen(vec, tamano_celda)
        mosaico.paste(img_v, (col_idx * ancho_img, 30))

    mosaico.save("mis_dibujos_mosaico.png")
    exportar_txt("mosaico.txt", vectores_creados)

    console.print(f"\n[EXITO] Imagen guardada como: mis_dibujos_mosaico.png")
    console.print(f"[EXITO] Archivo de datos guardado: mosaico.txt (Contiene {len(vectores_creados)} vectores)")

def main():
    console.print("=======================================")
    console.print(" HERRAMIENTA DE TRADUCCION Y DIBUJO 2D")
    console.print("=======================================")
    
    while True:
        console.print("\nSelecciona una opcion:")
        console.print("1. Generar mosaico PNG de reporte y archivo de datos")
        console.print("2. Traducir un vector existente a Imagen PNG")
        console.print("3. Abrir Dibujador Interactivo Multiple")
        console.print("4. Salir")
        
        opcion = Prompt.ask("Opcion", choices=["1", "2", "3", "4"])
        
        if opcion == "1":
            crear_mosaico_reporte()
        elif opcion == "2":
            traducir_vector_manual()
        elif opcion == "3":
            abrir_interfaz_buscaminas()
        elif opcion == "4":
            console.print("Cerrando.")
            break

if __name__ == "__main__":
    main()