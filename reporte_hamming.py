"""
hamming.py — Red Neuronal de Hamming
Carga los 30 vectores de mosaico.txt, entrena con los 10 limpios
y evalúa los 20 ruidosos. Genera reporte en consola y PNG.
"""

import math
import os
import random
from PIL import Image, ImageDraw, ImageFont


# ══════════════════════════════════════════════════════
#  UTILIDADES VISUALES
# ══════════════════════════════════════════════════════

def cargar_fuente(size: int):
    rutas = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ]
    for r in rutas:
        try:
            return ImageFont.truetype(r, size)
        except Exception:
            pass
    return ImageFont.load_default()


def pixel_art_consola(vector: list[int], ancho: int = 10) -> list[str]:
    lineas = []
    for i in range(0, len(vector), ancho):
        fila = vector[i:i + ancho]
        lineas.append("".join("██" if v == 1 else "  " for v in fila))
    return lineas


def distancia_hamming(a: list[int], b: list[int]) -> int:
    return sum(x != y for x, y in zip(a, b))


# ══════════════════════════════════════════════════════
#  CARGA DE DATOS
# ══════════════════════════════════════════════════════

def cargar_mosaico(path: str = "mosaico.txt") -> list[list[int]]:
    vectores = []
    with open(path, "r") as f:
        lineas = f.read().splitlines()
        
    # Validamos que al menos tenga el encabezado
    if len(lineas) < 3:
        raise ValueError("El archivo no tiene el formato correcto.")
        
    num_vecs = int(lineas[0])
    
    # Empezamos a iterar desde la línea 3 (donde realmente empiezan los vectores)
    for linea in lineas[3:3 + num_vecs]:
        linea = linea.strip()
        if not linea:
            continue
        vec = list(map(int, linea.split()))
        assert len(vec) == 100, f"Longitud inválida: {len(vec)}"
        assert all(v in (-1, 1) for v in vec), "Valores no bipolares"
        vectores.append(vec)
        
    return vectores


# ══════════════════════════════════════════════════════
#  RED DE HAMMING
# ══════════════════════════════════════════════════════

class RedHamming:
    def __init__(self, patrones: list[list[int]]):
        self.patrones = patrones          # p patrones de longitud n
        self.p = len(patrones)            # número de patrones
        self.n = len(patrones[0])         # longitud de cada patrón
        self.W = self._calcular_pesos()   # matriz p × n
        self.b = self.n / 2              # bias
        self.epsilon = 1 / (self.n - 1)  # inhibición lateral

    # ── Etapa 1: Pesos ─────────────────────────────
    def _calcular_pesos(self) -> list[list[float]]:
        return [[x / 2 for x in patron] for patron in self.patrones]

    # ── Etapa 1: Feedforward ───────────────────────
    def feedforward(self, Vp: list[int]) -> list[float]:
        """Calcula a(0) = ft( (1/n)·W·Vp + b )"""
        a0 = []
        for fila in self.W:
            dot = sum(w * v for w, v in zip(fila, Vp))
            val = (1 / self.n) * dot + self.b
            val = max(0.0, min(float(self.n), val))   # ReLU truncada a [0, n]
            a0.append(val)
        return a0

    # ── Etapa 2: MAXNET ────────────────────────────
    def maxnet(self, a0: list[float], max_iter: int = 200) -> tuple[list[float], list[list[float]], int]:
        """
        Competencia lateral hasta que solo quede una neurona activa.
        Devuelve (a_final, historial_iteraciones, num_iteraciones)
        """
        a = list(a0)
        historial = [list(a)]

        for it in range(1, max_iter + 1):
            a_new = []
            for i in range(self.p):
                suma_otros = sum(a[j] for j in range(self.p) if j != i)
                val = a[i] - self.epsilon * suma_otros
                a_new.append(max(0.0, val))

            historial.append(list(a_new))
            activas = sum(1 for v in a_new if v > 1e-9)

            if activas <= 1 or all(abs(a_new[i] - a[i]) < 1e-10 for i in range(self.p)):
                return a_new, historial, it

            a = a_new

        return a, historial, max_iter

    # ── Clasificación completa ─────────────────────
    def clasificar(self, Vp: list[int]) -> dict:
        a0         = self.feedforward(Vp)
        a_final, historial, iteraciones = self.maxnet(a0)

        ganadora   = a_final.index(max(a_final))
        activas    = sum(1 for v in a_final if v > 1e-9)
        hamming    = distancia_hamming(Vp, self.patrones[ganadora])

        if activas == 1 and hamming == 0:
            diagnostico = "ÉXITO"
        elif activas == 1 and hamming > 0:
            diagnostico = "FALLO"
        else:
            diagnostico = "NO ESTABLE"

        return {
            "a0":          a0,
            "a_final":     a_final,
            "historial":   historial,
            "iteraciones": iteraciones,
            "ganadora":    ganadora,
            "hamming":     hamming,
            "diagnostico": diagnostico,
        }


# ══════════════════════════════════════════════════════
#  REPORTE EN CONSOLA
# ══════════════════════════════════════════════════════

COLORES = {
    "ÉXITO":      "\033[92m",   # verde
    "FALLO":      "\033[91m",   # rojo
    "NO ESTABLE": "\033[93m",   # amarillo
    "RESET":      "\033[0m",
}


def imprimir_resultado(digito: int, grupo: str, res: dict):
    col = COLORES.get(res["diagnostico"], "")
    rst = COLORES["RESET"]
    print(f"\n  Dígito {digito:>2} [{grupo}]")
    print(f"    a(0)       = [{', '.join(f'{v:.3f}' for v in res['a0'])}]")
    print(f"    Iteraciones= {res['iteraciones']}")
    print(f"    Ganadora   = neurona {res['ganadora']} → Dígito {res['ganadora']}")
    print(f"    Hamming    = {res['hamming']}")
    print(f"    Resultado  = {col}{res['diagnostico']}{rst}")


def imprimir_historial(res: dict):
    print("    Historial MAXNET:")
    for t, a in enumerate(res["historial"]):
        print(f"      a({t}) = [{', '.join(f'{v:.4f}' for v in a)}]")


# ══════════════════════════════════════════════════════
#  REPORTE PNG
# ══════════════════════════════════════════════════════

def generar_reporte_png(
    red: RedHamming,
    vectores_prueba: list[list[int]],
    resultados: list[dict],
    etiquetas: list[str],
    path: str = "reporte_hamming.png",
):
    """Genera imagen con pares original/recuperado y diagnóstico."""
    CELL    = 10
    PAD     = 6
    COL_W   = 10 * CELL * 2 + PAD * 4 + 60   # original + recuperado
    ROW_H   = 10 * CELL + PAD * 2 + 50
    COLS    = 5
    ROWS    = math.ceil(len(vectores_prueba) / COLS)
    MARGIN  = 20
    HEADER  = 50

    img_w = MARGIN * 2 + COLS * COL_W
    img_h = MARGIN * 2 + HEADER + ROWS * ROW_H

    img  = Image.new("RGB", (img_w, img_h), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    f12 = cargar_fuente(12)
    f14 = cargar_fuente(14)
    f16 = cargar_fuente(16)

    draw.text(
        (MARGIN, MARGIN),
        "Reporte Red de Hamming — Evaluación de Vectores Ruidosos",
        fill=(20, 20, 80), font=f16,
    )

    COLOR_DIAG = {"ÉXITO": (20, 140, 20), "FALLO": (180, 20, 20), "NO ESTABLE": (180, 130, 0)}

    for idx, (vp, res, etq) in enumerate(zip(vectores_prueba, resultados, etiquetas)):
        col_i = idx % COLS
        row_i = idx // COLS
        x0 = MARGIN + col_i * COL_W
        y0 = MARGIN + HEADER + row_i * ROW_H

        # Encabezado
        draw.text((x0, y0), etq, fill=(60, 60, 60), font=f12)
        draw.text(
            (x0, y0 + 14),
            f"→ Dígito {res['ganadora']}  H={res['hamming']}  {res['diagnostico']}",
            fill=COLOR_DIAG.get(res["diagnostico"], (0, 0, 0)),
            font=f12,
        )

        y_px = y0 + 30

        # Dibujar vector de prueba (izquierda)
        for i, val in enumerate(vp):
            r_i, c_i = divmod(i, 10)
            px = x0 + c_i * CELL
            py = y_px + r_i * CELL
            fill = (20, 20, 20) if val == 1 else (215, 215, 215)
            draw.rectangle([px, py, px + CELL - 2, py + CELL - 2], fill=fill)

        # Separador
        sep_x = x0 + 10 * CELL + PAD
        draw.line([(sep_x, y_px), (sep_x, y_px + 10 * CELL)], fill=(150, 150, 150), width=1)

        # Dibujar patrón recuperado (derecha)
        recuperado = red.patrones[res["ganadora"]]
        x1 = sep_x + PAD
        for i, val in enumerate(recuperado):
            r_i, c_i = divmod(i, 10)
            px = x1 + c_i * CELL
            py = y_px + r_i * CELL
            fill = (20, 20, 20) if val == 1 else (215, 215, 215)
            draw.rectangle([px, py, px + CELL - 2, py + CELL - 2], fill=fill)

        # Mini etiquetas
        draw.text((x0,  y_px + 10 * CELL + 2), "Entrada",    fill=(80, 80, 80), font=f12)
        draw.text((x1,  y_px + 10 * CELL + 2), "Recuperado", fill=(80, 80, 80), font=f12)

    img.save(path)
    print(f"\n  ✔ Reporte PNG guardado: {path}")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("       RED NEURONAL DE HAMMING — EVALUACIÓN")
    print("=" * 60)

    # Cargar vectores
    if not os.path.exists("mosaico.txt"):
        print("\n  ✗ No se encontró mosaico.txt. Ejecuta primero dibujador.py")
        return

    todos    = cargar_mosaico("mosaico.txt")
    limpios  = todos[0:10]
    ruido10  = todos[10:20]
    ruido35  = todos[20:30]

    print(f"\n  Vectores cargados: {len(todos)} (10 limpios, 10 @ 10%, 10 @ 35%)")
    print(f"  Longitud de cada vector: {len(limpios[0])}")

    # Crear red
    red = RedHamming(limpios)
    print(f"\n  Red creada:")
    print(f"    Patrones almacenados (p) = {red.p}")
    print(f"    Longitud vectores    (n) = {red.n}")
    print(f"    Bias  b = n/2           = {red.b}")
    print(f"    Epsilon ε = 1/(n-1)     = {red.epsilon:.6f}")

    # Imprimir matriz W resumida
    print(f"\n  Matriz de pesos W ({red.p} × {red.n}) — primeros 5 valores por fila:")
    for i, fila in enumerate(red.W):
        muestra = [f"{v:5.2f}" for v in fila[:5]]
        print(f"    Fila {i} (Dígito {i}): [{', '.join(muestra)}, ...]")

    # ── Evaluar ruido 10% ──────────────────────────
    print("\n" + "=" * 60)
    print("  PRUEBA 1: Vectores con 10% de ruido")
    print("=" * 60)

    res_10 = []
    for d in range(10):
        res = red.clasificar(ruido10[d])
        res_10.append(res)
        imprimir_resultado(d, "10%", res)

    exitos_10 = sum(1 for r in res_10 if r["diagnostico"] == "ÉXITO")
    print(f"\n  Resumen 10%: {exitos_10}/10 éxitos")

    # ── Evaluar ruido 35% ──────────────────────────
    print("\n" + "=" * 60)
    print("  PRUEBA 2: Vectores con 35% de ruido")
    print("=" * 60)

    res_35 = []
    for d in range(10):
        res = red.clasificar(ruido35[d])
        res_35.append(res)
        imprimir_resultado(d, "35%", res)

    exitos_35 = sum(1 for r in res_35 if r["diagnostico"] == "ÉXITO")
    print(f"\n  Resumen 35%: {exitos_35}/10 éxitos")

    # ── Tabla final ────────────────────────────────
    print("\n" + "=" * 60)
    print("  TABLA RESUMEN FINAL")
    print("=" * 60)
    print(f"  {'Dígito':>6} | {'10% Ruido':^20} | {'35% Ruido':^20}")
    print("  " + "-" * 52)
    for d in range(10):
        r10 = res_10[d]
        r35 = res_35[d]
        col10 = "✔" if r10["diagnostico"] == "ÉXITO" else "✘"
        col35 = "✔" if r35["diagnostico"] == "ÉXITO" else "✘"
        print(
            f"  {d:>6} | {col10} → Dígito {r10['ganadora']} (H={r10['hamming']:>3}) "
            f"     | {col35} → Dígito {r35['ganadora']} (H={r35['hamming']:>3})"
        )
    print("  " + "-" * 52)
    print(f"  {'Total':>6} | {exitos_10}/10 éxitos{' '*12} | {exitos_35}/10 éxitos")

    # ── Reporte PNG ────────────────────────────────
    print("\n  Generando reporte PNG...")
    vp_todos   = ruido10 + ruido35
    res_todos  = res_10  + res_35
    etq_todos  = [f"D{d} — 10%" for d in range(10)] + [f"D{d} — 35%" for d in range(10)]
    generar_reporte_png(red, vp_todos, res_todos, etq_todos)

    print("\n¡Evaluación completa!")
    print("  • reporte_hamming.png → imagen con pares entrada/recuperado")


if __name__ == "__main__":
    main()
