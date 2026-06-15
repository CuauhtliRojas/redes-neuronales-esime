import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import simpledialog
from rich.console import Console
from rich.prompt import Prompt

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

def crear_mosaico_reporte():
    console.print("\nGenerando mosaico de 30 imagenes...")
    limpios = [texto_a_vector(num) for num in NUMEROS_ASCII]
    ruido_10 = [aplicar_ruido(v, 10) for v in limpios]
    ruido_35 = [aplicar_ruido(v, 35) for v in limpios]
    
    tamano_celda = 20
    ancho_img = 10 * tamano_celda
    alto_img = 10 * tamano_celda
    
    mosaico = Image.new('RGB', (10 * ancho_img, 3 * alto_img), color='white')
    
    for col in range(10):
        img_limpia = vector_a_imagen(limpios[col], tamano_celda)
        mosaico.paste(img_limpia, (col * ancho_img, 0))
        
        img_10 = vector_a_imagen(ruido_10[col], tamano_celda)
        mosaico.paste(img_10, (col * ancho_img, alto_img))
        
        img_35 = vector_a_imagen(ruido_35[col], tamano_celda)
        mosaico.paste(img_35, (col * ancho_img, 2 * alto_img))
        
    nombre_archivo = "reporte_vectores_mosaico.png"
    mosaico.save(nombre_archivo)
    console.print(f"Reporte exportado como imagen: {nombre_archivo}")

def traducir_vector_manual():
    console.print("\n--- TRADUCTOR: VECTOR -> IMAGEN ---")
    entrada = Prompt.ask("Ingresa los 100 numeros del vector (separados por espacio)")
    try:
        vec = np.array([int(x) for x in entrada.split()])
        if len(vec) != 100:
            console.print("Error: El vector debe tener exactamente 100 elementos.")
            return
            
        img = vector_a_imagen(vec)
        nombre = Prompt.ask("Nombre para guardar la imagen (ej. prueba1.png)")
        if not nombre.endswith(".png"): nombre += ".png"
        img.save(nombre)
        console.print(f"Imagen guardada exitosamente como {nombre}")
        
    except ValueError:
        console.print("Error: Asegurate de ingresar solo numeros separados por espacio.")

def abrir_interfaz_buscaminas():
    """Abre una ventana interactiva usando tkinter para dibujar la matriz."""
    console.print("\nAbriendo interfaz de dibujo. Revisa las ventanas de tu computadora...")
    
    ventana = tk.Tk()
    ventana.title("Traductor 10x10 - ESIME")
    ventana.resizable(False, False)
    
    # Estado inicial: vector de 100 elementos en -1 (blanco)
    estado_vector = [-1] * 100
    botones = []

    def toggle_pixel(idx):
        # Cambia de 1 a -1 o de -1 a 1
        estado_vector[idx] *= -1
        color = "black" if estado_vector[idx] == 1 else "white"
        botones[idx].config(bg=color)

    # Contenedor de la cuadricula
    frame_cuadricula = tk.Frame(ventana, bg="gray")
    frame_cuadricula.pack(padx=20, pady=20)

    # Crear los 100 botones
    for i in range(100):
        # width y height estan en unidades de texto, asi que se ajustan para parecer cuadrados
        btn = tk.Button(frame_cuadricula, width=4, height=2, bg="white", relief="raised",
                        command=lambda idx=i: toggle_pixel(idx))
        btn.grid(row=i//10, column=i%10, padx=1, pady=1)
        botones.append(btn)

    def guardar_y_cerrar():
        vector_final = np.array(estado_vector)
        
        # Ocultamos la cuadricula temporalmente para pedir el nombre
        nombre = simpledialog.askstring("Guardar Vector", "Nombre del archivo a exportar (ej. dibujo.png):", parent=ventana)
        
        if nombre:
            if not nombre.endswith(".png"): nombre += ".png"
            
            # Imprimir el vector en la consola
            print("\nVector Bipolar Generado:")
            print(" ".join(map(str, vector_final)))
            
            # Guardar la imagen
            img = vector_a_imagen(vector_final)
            img.save(nombre)
            print(f"Dibujo exportado exitosamente como {nombre}")
            
        ventana.destroy()

    btn_guardar = tk.Button(ventana, text="Guardar Vector y Exportar PNG", font=("Arial", 12, "bold"), command=guardar_y_cerrar)
    btn_guardar.pack(pady=10)

    # Iniciar el ciclo de la interfaz grafica
    ventana.mainloop()

def main():
    console.print("=======================================")
    console.print(" HERRAMIENTA DE TRADUCCION Y DIBUJO 2D")
    console.print("=======================================")
    
    while True:
        console.print("\nSelecciona una opcion:")
        console.print("1. Generar mosaico PNG de reporte (Numeros 0-9 con ruido)")
        console.print("2. Traducir un vector existente (100 numeros) a Imagen PNG")
        console.print("3. Abrir Dibujador Interactivo (Estilo Buscaminas)")
        console.print("4. Salir")
        
        opcion = Prompt.ask("Opcion", choices=["1", "2", "3", "4"])
        
        if opcion == "1":
            crear_mosaico_reporte()
        elif opcion == "2":
            traducir_vector_manual()
        elif opcion == "3":
            abrir_interfaz_buscaminas()
        elif opcion == "4":
            console.print("Cerrando traductor.")
            break

if __name__ == "__main__":
    main()