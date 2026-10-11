# =========================================================================
# INSTITUTO TECNOLÓGICO DE COSTA RICA
# CE1111: Análisis Numérico para Ingeniería
# Escuela de Matemática
#
# Portafolio Computacional - Bloque 3 (Parte 3)
# =========================================================================
from pathlib import Path
import csv
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from parte1 import metodo_QR, metodo_Jacobi


# Rutas relativas al archivi del script
DIR_SCRIPT = Path(__file__).resolve().parent
CARPETA_DATASET = DIR_SCRIPT / "dataset_att"
CARPETA_SALIDA = DIR_SCRIPT / "resultados_parte3"

# Metodo elegido
METODO = "QR"

# Parametros exigidos
ITER_MAX = 10**6
TOL = 10**-5

UMBRAL_VALOR_PROPIO = 1e-10 
VARIACION_OBJETIVO = 0.95 #Retener el 95% de la variacion acumulada

#Dimensiones originales de la imagenes
FILAS, COLUMNAS = 56, 46 #(56 filas x 46 columnas)
N_PIXELES = FILAS * COLUMNAS #2576

# Separacion entrenamientoprueba
PERSONAS = range(1, 41)
FOTOS_ENTRENAMIENTO = range(1, 10) #Fotos para entrenar 
FOTO_PRUEBA = 10 #Foto para probar


# Pasos del 1 al 3: Carga, preprocesamiento y vectorizacion de imagenes
def cargar_imagen(ruta):
    #Carga una imagen en escala de grises y la normaliza d [0,1]
    with Image.open(ruta) as im:
        im = im.convert("L")
        if im.size != (COLUMNAS, FILAS):
            raise ValueError(
                f"{ruta} tiene tamano {im.size}; se esperaba {(COLUMNAS, FILAS)}."
            )
        return np.asarray(im, dtype=np.float64) / 255.0


def vectorizar(imagen):
    # Convierte la imagen 56x46 en un vector de 2576 componentes por columna(Orden Fortran)
    return imagen.reshape(-1, order="F")


def cargar_entrenamiento():
    # Carga las 360 imagenes de entrenamiento 
    # Retorna
    # -S con cada imagen vectorizada como columna
    # - identificador de persona por columna
    #ruta del archivo original de cada imagene
    columnas, etiquetas, rutas = [], [], []
    for persona in PERSONAS:
        for foto in FOTOS_ENTRENAMIENTO:
            ruta = CARPETA_DATASET / f"s{persona}" / f"{foto}.png"
            if not ruta.exists():
                raise FileNotFoundError(f"No se encontro la imagen: {ruta}")
            columnas.append(vectorizar(cargar_imagen(ruta)))
            etiquetas.append(persona)
            rutas.append(ruta)
    # Orden de las columnas de S:persona 1(fotos 1...9)... persona 40(fotos 1...9)
    return np.column_stack(columnas), np.asarray(etiquetas), rutas


# Valores y vectores propios de G = A @ A.T
def calcular_eigenfaces(A):
    #Construye G = A @ A.T (2576 x 2576) y calcula sus valores y vectores propios con el metodo QR.
    #Retorno: lambdas,eigenfaces, error, iteraciones, convergencia y tiuempo.
    
    #Construccion de la matriz
    G = A @ A.T
    print(f"  Tamano de A.T @ A: {G.shape}")
    print(f"  Ejecutando {METODO} con iterMax = {ITER_MAX} y tol = {TOL}...")

    #Llamada al metodo elegido
    t0 = time.time()
    if METODO.strip().lower() == "qr":
        lambdas, V, error, iteraciones, convergio = metodo_QR(G, ITER_MAX, TOL)
    elif METODO.strip().lower() == "jacobi":
        lambdas, V, error, iteraciones, convergio = metodo_Jacobi(G, ITER_MAX, TOL)
    else:
        raise ValueError('METODO debe ser "QR" o "Jacobi".')
    tiempo = time.time() - t0

    print(f"  {METODO} termino en {tiempo:.1f} s | iters = {iteraciones} | "
          f"erk = {error:.3e} | conv = {convergio}")
    #Aplanamiento de valores propios(ya que pueden venir como columna)
    lambdas = np.asarray(lambdas).reshape(-1)
    V = np.asarray(V)

    # Valores propios en Orden decreciente
    orden = np.argsort(lambdas)[::-1]
    lambdas, V = lambdas[orden], V[:, orden]

    #Descartar valores propios nulos numericamente
    umbral = UMBRAL_VALOR_PROPIO * lambdas[0] if lambdas[0] > 0 else 0.0
    positivos = lambdas > umbral
    lambdas, V = lambdas[positivos], V[:, positivos]

    if lambdas.size == 0:
        raise RuntimeError("No se encontraron valores propios positivos.")
    # Vectores propios de G son caras base en el espacio de pixeles (cada uno con 2576 componentes)
    eigenfaces = V
    #Normalizacion de cada cara base
    normas = np.linalg.norm(eigenfaces, axis=0)
    eigenfaces = eigenfaces / np.where(normas > 0, normas, 1.0)[None, :]

    return lambdas, eigenfaces, error, iteraciones, convergio, tiempo


# Guardado de imagenes
def guardar_imagen_vector(vector, ruta):
    """Guarda un vector de 2576 componentes como imagen 56x46 en escala de grises."""
    imagen = np.asarray(vector).reshape((FILAS, COLUMNAS), order="F")
    imagen = np.clip(imagen, 0, 1)
    plt.imsave(ruta, imagen, cmap="gray", vmin=0, vmax=1)


def guardar_figura_variacion_acumulada(acumulada, k, ruta):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.arange(1, len(acumulada) + 1), acumulada,
            linewidth=1.6, color="#1f77b4", label="Variacion acumulada")
    ax.axhline(VARIACION_OBJETIVO, color="red", linestyle="--",
               label=f"{int(VARIACION_OBJETIVO*100)}% de la variacion")
    ax.axvline(k, color="green", linestyle="--", label=f"k = {k}")
    ax.set_xlabel("Numero de caras base")
    ax.set_ylabel("Variacion acumulada")
    ax.set_title("Variacion acumulada vs numero de caras base")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)


def guardar_figura_caras_base(eigenfaces, ruta):
    """Genera la figura 2x3 con las 6 primeras caras base."""
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.5))
    axes = axes.ravel()
    for j in range(6):
        ef = eigenfaces[:, j].copy()
        ef = (ef - ef.min()) / (ef.max() - ef.min() + 1e-15)
        axes[j].imshow(ef.reshape((FILAS, COLUMNAS), order="F"), cmap="gray")
        axes[j].set_title(f"Cara base {j+1}")
        axes[j].axis("off")
    fig.suptitle("Primeras seis caras base", fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Pasos 7-8: Proyeccion y calsificacion por vecino mas cercano
def predecir(Uk, media, S, etiquetas, rutas_entrenamiento):
    # Para cada imagen se prueba 
        # - vectorizar y centrar
        # - proyectar sobre Uk
        # - distancia euclidiana contra las 360 coordenadas de entrenamiento
        # - etiqueta del vecino mas cercano

    # Proyeccion de todas las imagenes de entrenamiento una unica vez
    coordenadas_entrenamiento = Uk.T @ (S - media[:, None])
    predicciones = []

    for persona_real in PERSONAS:
        ruta_prueba = CARPETA_DATASET / f"s{persona_real}" / f"{FOTO_PRUEBA}.png"
        if not ruta_prueba.exists():
            raise FileNotFoundError(f"No se encontro: {ruta_prueba}")

        # 1.Cargar y centrar imagen de prueba
        vector_prueba = vectorizar(cargar_imagen(ruta_prueba))
        centrada = vector_prueba - media

        # 2.Proyectar sobre las k caras base
        coordenada = Uk.T @ centrada

        # 3.Distancias euclidianas a las 360 imagenes conocidas
        distancias = np.linalg.norm(
            coordenadas_entrenamiento - coordenada[:, None], axis=0
        )
        # 4.Vecino mas cercano
        indice = int(np.argmin(distancias))
        persona_predicha = int(etiquetas[indice])

        predicciones.append({
            "foto_prueba": f"s{persona_real}/10.png",
            "persona_real": persona_real,
            "persona_predicha": persona_predicha,
            "correcta": "Si" if persona_real == persona_predicha else "No",
            "distancia": float(distancias[indice]),
            "foto_entrenamiento_mas_cercana": rutas_entrenamiento[indice].name,
            "persona_entrenamiento": int(etiquetas[indice]),
        })

    return predicciones

# Programa principal
def main():
    # Verificacion basica de la ruta del dataset
    if not CARPETA_DATASET.is_dir():
        raise FileNotFoundError(
            f"No se encontro la carpeta '{CARPETA_DATASET}'."
        )
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    # Pasos 1-4: Cargar imagenes, vectorizar , promediar y centrar
    print("=== Cargando 360 imagenes de entrenamiento ===")
    S, etiquetas, rutas_entrenamiento = cargar_entrenamiento()
    media = np.mean(S, axis=1)       # rostro promedio
    A = S - media[:, None]           # imagenes centradas
    print(f"  S: {S.shape}   A: {A.shape}")

    # Paso 5: valores y vectores propios de G
    print("=== Calculando eigenfaces ===")
    lambdas, eigenfaces, error, iteraciones, convergio, tiempo_qr = calcular_eigenfaces(A)
    n_eigen = len(lambdas)

    # Paso 6: elegir k segun variacion acumulada
    acumulada = np.cumsum(lambdas) / np.sum(lambdas)
    k = int(np.searchsorted(acumulada, VARIACION_OBJETIVO) + 1)
    k = min(k, len(acumulada))
    Uk = eigenfaces[:, :k]
    proporcion = float(np.sum(lambdas[:k]) / np.sum(lambdas))
    # Verificacion de ortogonalidad de las caras base retenida
    ortogonalidad = float(np.linalg.norm(Uk.T @ Uk - np.eye(k), ord="fro"))

    print(f"  r = {n_eigen}   k = {k}   variacion = {proporcion*100:.2f}%")
    print(f"  ||Uk^T Uk - I||_F = {ortogonalidad:.3e}")

    # Guardar modelo
    np.savez(
        CARPETA_SALIDA / "modelo_eigenfaces.npz",
        media=media, Uk=Uk, lambdas=lambdas,
        etiquetas=etiquetas, coordenadas_entrenamiento=Uk.T @ A,
    )

    # Guardar imagenes
    guardar_imagen_vector(media, CARPETA_SALIDA / "rostro_medio.png")
    for j in range(min(6, eigenfaces.shape[1])):
        ef = eigenfaces[:, j]
        # Reescale para visualizar
        ef_norm = (ef - ef.min()) / (ef.max() - ef.min() + 1e-15)
        guardar_imagen_vector(ef_norm, CARPETA_SALIDA / f"eigenface_{j+1}.png")
    guardar_figura_caras_base(eigenfaces, CARPETA_SALIDA / "caras_base_1_a_6.png")
    guardar_figura_variacion_acumulada(acumulada, k,
                                       CARPETA_SALIDA / "variacion_acumulada.png")

    # Pasos 7-8:proyectar las 40 imagenes de prueba y clasificar
    print("=== Evaluando las 40 imagenes de prueba ===")
    predicciones = predecir(Uk, media, S, etiquetas, rutas_entrenamiento)
    aciertos = sum(p["correcta"] == "Si" for p in predicciones)
    exactitud = 100.0 * aciertos / len(predicciones)

    # Guardar tabla de predicciones
    csv_path = CARPETA_SALIDA / "predicciones.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        campos = ["foto_prueba", "persona_real", "persona_predicha",
                  "correcta", "distancia", "foto_entrenamiento_mas_cercana"]
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predicciones)

    # Resumen en consola
    print("\n=== RESUMEN ===")
    print(f"  Metodo              : {METODO}")
    print(f"  Iteraciones         : {iteraciones}")
    print(f"  Convergio (conv)    : {convergio}")
    print(f"  Error final (erk)   : {error:.3e}")
    print(f"  Tiempo de QR        : {tiempo_qr:.1f} s")
    print(f"  r (valores propios) : {n_eigen}")
    print(f"  k (caras base)      : {k}")
    print(f"  Variacion           : {proporcion*100:.2f}%")
    print(f"  Aciertos            : {aciertos} / {len(predicciones)}")
    print(f"  Exactitud           : {exactitud:.2f}%")

    print("\n=== ARCHIVOS GENERADOS ===")
    print(f"  CSV                 : {csv_path.resolve()}")
    print(f"  Modelo              : {(CARPETA_SALIDA / 'modelo_eigenfaces.npz').resolve()}")
    print(f"  Rostro promedio     : {(CARPETA_SALIDA / 'rostro_medio.png').resolve()}")
    print(f"  Caras base 1-6      : {(CARPETA_SALIDA / 'eigenface_1.png').resolve()} ...")
    print(f"  Figura caras base   : {(CARPETA_SALIDA / 'caras_base_1_a_6.png').resolve()}")
    print(f"  Variacion acumulada : {(CARPETA_SALIDA / 'variacion_acumulada.png').resolve()}")

if __name__ == "__main__":
    main()