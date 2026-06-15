import numpy as np
from rich.console import Console
from rich.panel import Panel

# Importamos tu motor matemático
from redes.hopfield import RedHopfield

console = Console()

def matriz_a_vector(texto_matriz):
    """Limpia los saltos de línea de la cuadrícula visual y genera el vector 1D de 121 elementos."""
    numeros = texto_matriz.replace('\n', ' ').split()
    return np.array([int(n) for n in numeros])

def main():
    console.print(Panel.fit("[bold green]Prueba de Cuadrícula 11x11 (121 neuronas)[/bold green]"))

    # PATRÓN 1: Línea Vertical
    cuadricula_1 = """
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    """

    # PATRÓN 2: Línea Horizontal
    cuadricula_2 = """
    -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
     1  1  1  1  1  1  1  1  1  1  1
    -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
    """

    # VECTOR DE PRUEBA: Línea vertical con dos errores (ruido) en la esquina superior izquierda
    cuadricula_prueba = """
     1  1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1
    """

    # Convertir textos a vectores de numpy
    p1 = matriz_a_vector(cuadricula_1)
    p2 = matriz_a_vector(cuadricula_2)
    prueba = matriz_a_vector(cuadricula_prueba)

    console.print(f"Dimensiones de los vectores: P1({len(p1)}), P2({len(p2)}), Prueba({len(prueba)})")

    # Inicializar y entrenar la red
    console.print("\n[bold cyan]Entrenando red con los 2 patrones...[/bold cyan]")
    red = RedHopfield(tamano=121, tipo_func='bipolar')
    red.entrenar([p1, p2])
    console.print("✅ Red entrenada. Matriz de pesos (121x121) calculada internamente.")

    # Fase de prueba
    console.print("\n[bold yellow]--- Evaluando el vector de prueba con ruido ---[/bold yellow]")
    convergio, estado_final, iteraciones, dist_final, historial = red.predecir(prueba)

    if convergio:
        if np.array_equal(estado_final, p1):
            asociacion = "Patrón 1 (Línea Vertical)"
        elif np.array_equal(estado_final, p2):
            asociacion = "Patrón 2 (Línea Horizontal)"
        else:
            asociacion = "Ninguno (Convergió a un patrón desconocido)"

        console.print(f"\n[bold green]¡Convergencia en la iteración {iteraciones}![/bold green]")
        console.print(f"La red corrigió {dist_final} píxeles de ruido.")
        console.print(f"[bold magenta]Resultado Final:[/bold magenta] Asoció con [bold]{asociacion}[/bold]")
    else:
        console.print("\n[bold red]La red entró en un ciclo y no convergió.[/bold red]")

if __name__ == "__main__":
    main()