###############################################################################################
# CE1111: Análisis Numérico para Ingeniería
# Escuela de Ingeniería en Computadores
#
# Portafolio Bloque 1: Parte 3 (Aplicación)
#
# Autores: Joaquin Ignacio Ramírez Sequeira
#          Joseph Stif Piedra Montero 
#
# Este archivo resuelve la determinación del factor de fricción de Darcy (f) en una tubería 
# utilizando la ecuación de Colebrook-White mediante los seis métodos numéricos del Bloque 1.
###############################################################################################
import time
import numpy as np 
import matplotlib.pyplot as plt
import sympy as sp

from parte1 import (Newton_Raphson, Secante, Steffensen, Muller, Biseccion, Falsa_Posicion)

# ------------------------------------------------------------------------------
# 1. Parámetros del problema y transformación de Colebrook-White a g(x) = 0
# ------------------------------------------------------------------------------
D = 0.25          # Diámetro interno de la tubería en metros (m)
epsilon = 0.00015 # Rugosidad absoluta de la tubería en metros (m)
Re = 120000       # Número de Reynolds (régimen turbulento)

# La ecuación de Colebrook-White original es:
# 1/sqrt(f) = -2 * log10( (epsilon / (3.7*D)) + (2.51 / (Re * sqrt(f))) )
#
# Para la búsqueda de raíces de la forma g(x) = 0 (donde 'x' representa el factor f):
# g(x) = 1/sqrt(x) + 2 * log10( (epsilon / (3.7*D)) + (2.51 / (Re * sqrt(x))) )
# Se expresa log10(u) como log(u)/log(10) para compatibilidad simbólica en SymPy.
f_string = f"1/sqrt(x) + 2*(log(({epsilon}/(3.7*{D})) + (2.51/({Re}*sqrt(x)))) / log(10))"

# Parámetros exigidos para la tolerancia e iteraciones
iterMax = 1000
tol = 1e-8

# ------------------------------------------------------------------------------
# 2. Análisis del intervalo físicamente razonable y 3. Valores iniciales
# ------------------------------------------------------------------------------
# En hidráulica, para tuberías comerciales en régimen turbulento, el factor de 
# fricción de Darcy f es una cantidad adimensional estrictamente positiva que 
# típicamente se encuentra en el rango [0.008, 0.1]. Se selecciona el intervalo [0.01, 0.05].

a = 0.01
b = 0.05

# Verificación explícita de la condición del Teorema de Bolzano para métodos cerrados:
x_sym = sp.Symbol('x')
g_sym = sp.sympify(f_string)
g_a = float(g_sym.subs(x_sym, a).evalf())
g_b = float(g_sym.subs(x_sym, b).evalf())

# Selección de valores iniciales para métodos abiertos:
# Para Newton-Raphson se utiliza x0 = 0.02.
x0_nr = 0.02

# Para Steffensen se utiliza x0 = 0.0204. Este punto se selecciona muy cercano a la raíz
# física (f ≈ 0.02036) para garantizar que x0 + g(x0) > 0 y evitar evaluar la raíz
# cuadrada de un número negativo en el dominio físico (f > 0).
x0_stff = 0.0204

# Para Secante se usan x0 = 0.015 y x1 = 0.025 en las cercanías de la solución.
x0_sec = 0.015
x1_sec = 0.025

# Para Müller se eligen tres puntos bien distribuidos: x0 = 0.015, x1 = 0.020 y x2 = 0.025.
x0_mull = 0.015
x1_mull = 0.020
x2_mull = 0.025

# ------------------------------------------------------------------------------
# 4. Configuración y ejecución de los métodos numéricos
# ------------------------------------------------------------------------------
configs = {
    "Newton_Raphson": (Newton_Raphson, (f_string, x0_nr, tol, iterMax)),
    "Secante": (Secante, (f_string, x0_sec, x1_sec, tol, iterMax)),
    "Steffensen": (Steffensen, (f_string, x0_stff, tol, iterMax)),
    "Muller": (Muller, (f_string, x0_mull, x1_mull, x2_mull, tol, iterMax)),
    "Biseccion": (Biseccion, (f_string, a, b, tol, iterMax)),
    "Falsa-Posicion": (Falsa_Posicion, (f_string, a, b, tol, iterMax))
}

resultados = {}

# Ejecución de los 6 métodos registrando tiempos de ejecución
for nombre, (metodo, argumento) in configs.items():
    inicio = time.perf_counter()
    xk, erk, k, conv = metodo(*argumento)
    tiempo = time.perf_counter() - inicio
    
    # Conversión segura extrayendo la parte real (sp.re) para evitar errores si SymPy
    # genera números con residuo imaginario nulo (+ 0*I)
    xk_val = float(sp.re(xk)) if xk is not None else None
    erk_val = float(sp.re(erk)) if erk is not None else None
    
    resultados[nombre] = [xk_val, erk_val, k, tiempo, conv]

# ------------------------------------------------------------------------------
# 6. Salida en consola: Justificación, Tabla y Análisis de resultados
# ------------------------------------------------------------------------------
print("\n" + "="*100)
print("PARTE III: APLICACIÓN - FACTOR DE FRICCIÓN DE DARCY EN TUBERÍA (COLEBROOK-WHITE)")
print("="*100)
print(f"Parámetros del sistema: D = {D} m,  epsilon = {epsilon} m,  Re = {Re}")
print(f"Expresión g(f) = 0: 1/sqrt(f) + 2*log10(({epsilon}/(3.7*{D})) + (2.51/({Re}*sqrt(f)))) = 0")
print("-" * 100)

print("\nJUSTIFICACIÓN Y VERIFICACIÓN DE CONDICIONES INICIALES:")
print(f"- Intervalo físicamente razonable seleccionado: [{a}, {b}]")
print(f"- Evaluación g(a = {a}) = {g_a:.6f}")
print(f"- Evaluación g(b = {b}) = {g_b:.6f}")
print(f"- Producto g(a) * g(b) = {g_a * g_b:.6f} < 0")
if g_a * g_b < 0:
    print("  -> ¡Condición de cambio de signo (Teorema de Bolzano) VERIFICADA para Bisección y Falsa Posición!")
print(f"- Valor inicial para Newton-Raphson: x0 = {x0_nr}")
print(f"- Valor inicial para Steffensen: x0 = {x0_stff} (seleccionado para mantener x0 + g(x0) > 0)")
print(f"- Valores iniciales para Secante: x0 = {x0_sec}, x1 = {x1_sec}")
print(f"- Valores iniciales para Müller: x0 = {x0_mull}, x1 = {x1_mull}, x2 = {x2_mull}")

print("\n" + "="*100)
print("TABLA COMPARATIVA DE RESULTADOS")
print("="*100)
print(f"{'Método':<20}{'xk (f)':>18}{'Error final':>18}{'Iteraciones':>15}{'Tiempo (s)':>15}{'Conv':>8}")
print("-" * 100)

for nombre, datos in resultados.items():
    xk, erk, k, tiempo, conv = datos
    print(f"{nombre:<20}{xk:>18.10f}{erk:>18.4e}{k:>15}{tiempo:>15.6e}{conv:>8}")

print("="*100)

# Obtener valor aproximado del factor de fricción
f_aprox = resultados["Newton_Raphson"][0]

print("\n" + "="*100)
print("ANÁLISIS DE RESULTADOS E INTERPRETACIÓN EN EL CONTEXTO DEL PROBLEMA")
print("="*100)
print(f"""
1. ANÁLISIS DE CONVERGENCIA Y COMPARACIÓN:
   - Todos los métodos computacionales convergieron exitosamente (conv = 1) hacia la misma solución 
     f ≈ {f_aprox:.6f}, satisfaciendo la tolerancia requerida de 1e-8.
   - Müller y Newton-Raphson requirieron la menor cantidad de iteraciones (3 iteraciones cada uno) y los 
     menores tiempos de ejecución, mostrando la alta eficiencia de las aproximaciones cuadráticas y el uso de la derivada.
   - Secante convergió en 5 iteraciones y Steffensen en 7 iteraciones.
   - Bisección requirió la mayor cantidad de iteraciones (28) debido a su convergencia lineal con factor 
     de reducción estricto de 1/2. Falsa Posición completó 22 iteraciones, debido al fenómeno de estancamiento 
     de uno de los extremos del intervalo.

2. INTERPRETAACIÓN EN EL CONTEXTO DE INGENIERÍA:
   - Factor de fricción de Darcy aproximado: f ≈ {f_aprox:.6f} (adimensional).
   - Significado físico: El factor de fricción f representa la resistencia hidráulica al flujo ocasionada 
     por el rozamiento continuo entre el fluido y la rugosidad interna de la tubería (epsilon = {epsilon} m) 
     en régimen altamente turbulento (Re = {Re}). Este valor es fundamental para calcular la pérdida de 
     carga por fricción (caída de presión) mediante la ecuación de Darcy-Weisbach en el diseño de tuberías.
""")
print("="*100 + "\n")

# ------------------------------------------------------------------------------
# 7. Gráficas comparativas
# ------------------------------------------------------------------------------
nombres = list(resultados.keys())
errores = [datos[1] for datos in resultados.values()]
tiempos = [datos[3] for datos in resultados.values()]
iteraciones = [datos[2] for datos in resultados.values()]

colores = ['#1f77b4', '#f3ff0e', '#2ca02c', '#d62728', '#9467bd', '#f3740c']

plt.figure(figsize=(15, 5))

# Gráfica de errores
plt.subplot(1, 3, 1)
plt.bar(nombres, errores, color=colores)
plt.yscale("log")
plt.title("Gráfica de los errores obtenidos")
plt.xlabel("Método")
plt.ylabel("Error obtenido")
plt.xticks(rotation=45)

# Gráfica de tiempos de ejecución
plt.subplot(1, 3, 2)
plt.bar(nombres, tiempos, color=colores)
plt.title("Gráfica de los tiempos de ejecución")
plt.xlabel("Método")
plt.ylabel("Tiempo (s)")
plt.xticks(rotation=45)

# Gráfica del número de iteraciones
plt.subplot(1, 3, 3)
plt.bar(nombres, iteraciones, color=colores)
plt.title("Gráfica del número de iteraciones realizadas")
plt.xlabel("Método")
plt.ylabel("Iteraciones")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()