"""
INSTITUTO TECNOLÓGICO DE COSTA RICA
CE-1111: Análisis Numérico para Ingeniería
Semestre II-2026
Proyecto de Investigación - Implementación de Métodos Numéricos para EDOs

Autores: Joseph Stif Piedra Montero
         Joaquin Ignacio Ramírez Sequeira

Módulo: metodos_edo.py
Descripción:
    Contiene la implementación explícita de los métodos numéricos de un paso
    para resolver el Problema de Cauchy (Problema de Valor Inicial):
        y'(t) = f(t, y(t)),   y(t0) = y0,   t en [a, b]
    utilizando una malla uniforme de n subintervalos con tamaño de paso:
        h = (b - a) / n
        t_j = a + j * h,   para j = 0, 1, ..., n

Métodos incluidos:
    1. euler_explicito : Método de Euler explícito (orden 1)
    2. heun_rk2        : Método Predictor-Corrector de Heun / Euler mejorado (orden 2)
    3. rk4_clasico     : Método de Runge-Kutta clásico de cuarto orden (orden 4)
"""

import numpy as np


def euler_explicito(f, a, b, y0, n):
    """
    Aproxima la solución del problema de Cauchy mediante el Método de Euler explícito.

    Fórmula iterativa:
        y_{k+1} = y_k + h * f(t_k, y_k),   para k = 0, 1, ..., n - 1

    Parámetros:
        f  (callable) : Función f(t, y) que define la EDO y' = f(t, y).
        a  (float)    : Tiempo inicial t0 del intervalo [a, b].
        b  (float)    : Tiempo final tf del intervalo [a, b].
        y0 (float)    : Condición inicial y(a) = y0.
        n  (int)      : Número de subintervalos (pasos de integración).

    Retorna:
        t (np.ndarray) : Vector de nodos temporales [t_0, t_1, ..., t_n] de tamaño (n + 1,).
        y (np.ndarray) : Vector de aproximaciones [y_0, y_1, ..., y_n] de tamaño (n + 1,).
    """
    # Cálculo del tamaño de paso uniforme h
    h = (b - a) / n

    # Construcción de la malla discreta de tiempos t_j = a + j*h
    t = np.linspace(a, b, n + 1)

    # Inicialización del arreglo de soluciones y asignación de la condición inicial
    y = np.zeros(n + 1, dtype=float)
    y[0] = y0

    # Proceso iterativo paso a paso desde k = 0 hasta n - 1 (criterio de parada: n pasos)
    for k in range(n):
        pendiente = f(t[k], y[k])
        y[k + 1] = y[k] + h * pendiente

    return t, y


def heun_rk2(f, a, b, y0, n):
    """
    Aproxima la solución del problema de Cauchy mediante el Método de Heun
    (Predictor-Corrector o Euler mejorado, Runge-Kutta de orden 2).

    Fórmula iterativa:
        K1 = f(t_k, y_k)
        K2 = f(t_k + h, y_k + h * K1)
        y_{k+1} = y_k + (h / 2) * (K1 + K2),   para k = 0, 1, ..., n - 1

    Parámetros:
        f  (callable) : Función f(t, y) que define la EDO y' = f(t, y).
        a  (float)    : Tiempo inicial t0 del intervalo [a, b].
        b  (float)    : Tiempo final tf del intervalo [a, b].
        y0 (float)    : Condición inicial y(a) = y0.
        n  (int)      : Número de subintervalos (pasos de integración).

    Retorna:
        t (np.ndarray) : Vector de nodos temporales [t_0, t_1, ..., t_n] de tamaño (n + 1,).
        y (np.ndarray) : Vector de aproximaciones [y_0, y_1, ..., y_n] de tamaño (n + 1,).
    """
    # Cálculo del tamaño de paso uniforme h
    h = (b - a) / n

    # Construcción de la malla discreta de tiempos
    t = np.linspace(a, b, n + 1)

    # Inicialización del arreglo de soluciones
    y = np.zeros(n + 1, dtype=float)
    y[0] = y0

    # Proceso iterativo paso a paso desde k = 0 hasta n - 1
    for k in range(n):
        # Pendiente al inicio del intervalo (predictor)
        K1 = f(t[k], y[k])
        # Pendiente estimada al final del intervalo (corrector)
        K2 = f(t[k] + h, y[k] + h * K1)
        # Actualización mediante promedio trapezoidal de pendientes
        y[k + 1] = y[k] + (h / 2.0) * (K1 + K2)

    return t, y


def rk4_clasico(f, a, b, y0, n):
    """
    Aproxima la solución del problema de Cauchy mediante el Método Clásico de
    Runge-Kutta de cuarto orden (RK4).

    Fórmula iterativa:
        K1 = f(t_k, y_k)
        K2 = f(t_k + h/2, y_k + (h/2) * K1)
        K3 = f(t_k + h/2, y_k + (h/2) * K2)
        K4 = f(t_k + h,   y_k + h * K3)
        y_{k+1} = y_k + (h / 6) * (K1 + 2*K2 + 2*K3 + K4),   para k = 0, 1, ..., n - 1

    Parámetros:
        f  (callable) : Función f(t, y) que define la EDO y' = f(t, y).
        a  (float)    : Tiempo inicial t0 del intervalo [a, b].
        b  (float)    : Tiempo final tf del intervalo [a, b].
        y0 (float)    : Condición inicial y(a) = y0.
        n  (int)      : Número de subintervalos (pasos de integración).

    Retorna:
        t (np.ndarray) : Vector de nodos temporales [t_0, t_1, ..., t_n] de tamaño (n + 1,).
        y (np.ndarray) : Vector de aproximaciones [y_0, y_1, ..., y_n] de tamaño (n + 1,).
    """
    # Cálculo del tamaño de paso uniforme h
    h = (b - a) / n

    # Construcción de la malla discreta de tiempos
    t = np.linspace(a, b, n + 1)

    # Inicialización del arreglo de soluciones
    y = np.zeros(n + 1, dtype=float)
    y[0] = y0

    # Proceso iterativo paso a paso desde k = 0 hasta n - 1
    for k in range(n):
        K1 = f(t[k], y[k])
        K2 = f(t[k] + 0.5 * h, y[k] + 0.5 * h * K1)
        K3 = f(t[k] + 0.5 * h, y[k] + 0.5 * h * K2)
        K4 = f(t[k] + h, y[k] + h * K3)

        # Combinación ponderada de las cuatro pendientes
        y[k + 1] = y[k] + (h / 6.0) * (K1 + 2.0 * K2 + 2.0 * K3 + K4)

    return t, y