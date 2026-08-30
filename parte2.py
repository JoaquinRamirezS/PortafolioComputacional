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

# Para Secante se utilizan los puntos x0 = 1 y x1 = 2,ya que ambos se encuentran 
# alrededor de la raíz positiva aproximada x= = 1.67.

x0_sec = 1
x1_sec = 2

# Para Muller alrededor de la raíz para construir la parábola que requiere el método.
# Por esta razón, se utilizan los puntos x0 = 1, x1 = 1.5 y x2 = 2.

x0_mull = 1
x1_mull = 1.5
x2_mull = 2

# Por último, para Bisección y Falsa Posición se busca un intervalo que contenga la 
# raíz y en cuyos puntos exista un cambio de signo al evaluar la función.Debido a esto 
# se eligió hel intervalo [1,2].

a = 1
b = 2

#----------------------------------------------------------------------------------------
# Configuración de los métodos
#----------------------------------------------------------------------------------------

# Se crea un diccionario donde se almacena cada método juntos con los argumentos que 
# necesita para su ejecución

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
# Se crea un diccionario vacío para almcenar los resultados que se otbienen de cada método
resultados = {}

#Se recorren todos los métodos
for nombre, (metodo,argumento) in configs.items():
    inicio = time.perf_counter()  # Se registra el tiempo antes de ejecutar el método
    xk, erk, k, conv = metodo(*argumento) # Se ejecuta el método utilizando los argumentos correspondientes
    tiempo = time.perf_counter() - inicio # Se calcula el tiempo que tardó en ejecutarse cada método
    resultados[nombre] = [xk, erk, k, tiempo, conv] # Se almacenan los resultados por método

#----------------------------------------------------------------------------------------
# Tabla comparativa
#----------------------------------------------------------------------------------------
print("\n")
print("="*100)
print("Tabla Comparativa")
print("="*100)

# Se imprimen los encabezados de las columnas
print(f"{'Metodo':<25}{'xk':>18}{'Error':>18}"
      f"{'Iteraciones':>15}{'Tiempo(s)':>15}{'Conv':>8}")
# Se recorren los resultados almacenados para mostrar la info de cada método
for nombre, datos in resultados.items():
    xk, erk, k, tiempo, conv = datos # Los datos almacenados se separan

    # Se imprime cada resultado
    print(f"{nombre:<25}{xk:>18.10f}{erk:>18.4e}"
      f"{k:>15}{tiempo:>15.6e}{conv:>8}")

print("="*100)

#----------------------------------------------------------------------------------------
# Gráficas comparativas 
#----------------------------------------------------------------------------------------
nombres = list(resultados.keys()) # Nombres de los métodos en el eje x
errores = [datos[1] for datos in resultados.values()] # Se extraen errores obtenidos por método
tiempos = [datos[3] for datos in resultados.values()] # Se extraen los tiempos de ejecución de cada método
iteraciones = [datos[2] for datos in resultados.values()] # Se extraen el número e iteraciones realizadas por método

# Colores para cada barra de las gráfica
colores = ['#1f77b4', "#f3ff0e", '#2ca02c', '#d62728', '#9467bd', "#f3740c"]

plt.figure(figsize =(15, 5))
#----------------------------------------------------------------------------------------
# Gráfica de los errores obtenidos
#----------------------------------------------------------------------------------------
plt.subplot(1, 3, 1) # Posición en la cuadrícula
plt.bar(nombres, errores, color=colores) # Gráfica de barras con el error final de cada método
plt.yscale("log") # Se usa una escala logaritmica 
plt.title("Gráfica de los errores obtenidos")
plt.xlabel("Método")
plt.ylabel("Error obtenido")
plt.xticks(rotation=45)

#----------------------------------------------------------------------------------------
# Gráfica de los tiempos de ejecución
#----------------------------------------------------------------------------------------
plt.subplot(1, 3, 2) # Se coloca en la segunda posición de la cuadrícula
plt.bar(nombres, tiempos, color=colores) # Se crea la gráfica de barras

plt.title("Gráfica de los tiempos de ejecución")
plt.xlabel("Método")
plt.ylabel("Tiempo(s)")
plt.xticks(rotation=45)

#----------------------------------------------------------------------------------------
# Gráfica del número de iteraciones
#----------------------------------------------------------------------------------------
plt.subplot(1, 3, 3) # Se coloca en tercera posición de la cuadrícula
plt.bar(nombres, iteraciones, color=colores) # Se crea la gráfica de barras


plt.title("Gráfica del núero de iteraciones realizadas")
plt.xlabel("Método")
plt.ylabel("Iteraciones")
plt.xticks(rotation=45)

plt.subplots_adjust(wspace=0.8)

plt.tight_layout()
plt.show()