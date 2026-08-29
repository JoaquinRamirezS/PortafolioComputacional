##############################################################################
# CE1111: Análisis Numérico para Ingeniería
# Escuela de Ingeniería en Computadores
#
# Portafolio Bloque 1:Parte 2
#
# Autores: Joaquin Ignacio Ramírez Sequeira
#          Joseph Stif Piedra Montero 
# Utilizar los seis métodos computacionales para aproximar una misma solución
# de la ecuación dada.
##############################################################################
import time
import numpy as np 
import matplotlib.pyplot as plt

from parte1 import (Newton_Raphson,Secante,Steffensen,Muller,Biseccion,Falsa_Posicion)

# Función en formato texto
f_string="log(x**2 + 1) - cos(x) - 4*x**2 + 10"

# Parmátros para utilizar en todos los métodos
iterMax = 1000
tol = 1e-8
# ------------------------------------------------------------------------------
# Selección de los valores iniciales 
# ------------------------------------------------------------------------------
# Primeramente se gráfico la función y se observó que hay dos soluciones.Hay una
# raíz negativa aproximadamente en x=-1.67 y una raíz positiva en x=1.67.Basado en
# esto se seleccionó la raíz positiva como punto de referncia y se seleccionaron 
# valores cercanos a esta raíz.

#------------------------------------------------------------------------------

# ##############################################################################
# Justificación por método
# ##############################################################################

# Para Newton-Raphson y Steffensen se utiliza x0 = 1.5, ya que solo se requiere 
# un único punto inicial y este se encuentra cercano a la raíz que se desea encontrar.

x0_nr_stff = 1.5

# Para Secante se utilizan los puntos x0 = 1 y x1 = 2,ya que ambos se encuentran alrededor
# de la raíz positiva aproximada x= = 1.67.

x0_sec = 1
x1_sec = 2

# Para Muller alrededor de la raíz para construir la parábola que requiere el método.Por 
# esta razón, se utilizan los puntos x0 = 1, x1 = 1.5 y x2 = 2.

x0_mull = 1
x1_mull = 1.5
x2_mull = 2

# Por último, para Bisección y Falsa Posición se busca un intervalo que contenga la raíz 
# y en cuyos puntos exista un cambio de signo al evaluar la función.Debido a esto se eligió
# el intervalo [1,2].

a = 1
b = 2

#----------------------------------------------------------------------------------------
# Configuración de los métodos
#----------------------------------------------------------------------------------------
configs = {
"Newton_Raphson":(Newton_Raphson,(f_string,x0_nr_stff,tol,iterMax)),
"Secante":(Secante,(f_string,x0_sec,x1_sec,tol,iterMax)),
"Steffensen":(Steffensen,(f_string,x0_nr_stff,tol,iterMax)),
"Muller":(Muller,(f_string,x0_mull,x1_mull,x2_mull,tol,iterMax)),
"Biseccion":(Biseccion,(f_string,a,b,tol,iterMax)),
"Falsa-Posicion":(Falsa_Posicion,(f_string,a,b,tol,iterMax)) }

#----------------------------------------------------------------------------------------
# Ejecucion de los métodos
#----------------------------------------------------------------------------------------

resultados = {}
for nombre, (metodo,argumento) in configs.items():
    inicio = time.perf_counter()
    xk, erk, k, conv = metodo(*argumento)
    tiempo = time.perf_counter() - inicio
    resultados[nombre] = [xk, erk, k, tiempo, conv]

#----------------------------------------------------------------------------------------
# Tabla comparativa
#----------------------------------------------------------------------------------------
print("\n")
print("="*100)
print("Tabla Comparativa")
print("="*100)

print(f"{'Metodo':<25}{'xk':>18}{'Error':>18}"
      f"{'Iteraciones':>15}{'Tiempo(s)':>15}{'Conv':>8}")

for nombre, datos in resultados.items():
    xk, erk, k, tiempo, conv = datos

    print(f"{nombre:<25}{xk:>18.10f}{erk:>18.4e}"
      f"{k:>15}{tiempo:>15.6e}{conv:>8}")

print("="*100)


