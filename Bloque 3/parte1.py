# =========================================================================
# INSTITUTO TECNOLÓGICO DE COSTA RICA
# CE1111: Análisis Numérico para Ingeniería
# Escuela de Matemática
#
# Portafolio Computacional - Bloque 3 (Parte 1)
# =========================================================================

import numpy as np


def metodo_QR(A, iterMax, tol):
    """
    Método QR para la aproximación de los valores y vectores propios 
    de una matriz simétrica semidefinida positiva.
    
    Parámetros:
    -----------
    A       : numpy.ndarray - Matriz cuadrada simétrica semidefinida positiva.
    iterMax : int           - Número máximo de iteraciones permitidas.
    tol     : float         - Tolerancia para el criterio de parada.
    
    Retorna:
    --------
    lambdas : numpy.ndarray - Vector columna con los valores propios ordenados de mayor a menor.
    V       : numpy.ndarray - Matriz con los vectores propios normalizados como columnas.
    erk     : float         - Error de la última iteración ||lambda^(k) - lambda^(k-1)||_2.
    k       : int           - Número de iteraciones ejecutadas.
    conv    : int           - 1 si el método convergió, 0 en caso contrario.
    """
    n = A.shape[0]

    # Inicialización 
    B = np.copy(A).astype(float)
    V = np.eye(n, dtype=float)
    conv = 0

    # Vector lambda inicial ordenado de mayor a menor
    lambda_prev = np.sort(np.diag(B))[::-1]
    erk = float("inf")
    idx_orden = np.arange(n)

    # Bucle iterativo
    for k in range(1, iterMax + 1):
        # Factorización QR de B
        Q, R = np.linalg.qr(B)

        # Fuerza diagonal de R positiva 
        diag_R = np.diag(R)
        signos = np.where(diag_R < 0, -1.0, 1.0)
        D = np.diag(signos)

        Q = Q @ D
        R = D @ R

        # Siguiente paso del método QR
        C = R @ Q

        # Acumulación de transformaciones ortogonales en V
        V = V @ Q

        # Extracción y ordenamiento decreciente de valores propios
        diag_C = np.diag(C)
        idx_orden = np.argsort(diag_C)[::-1]
        lambda_k = diag_C[idx_orden]

        # Cálculo del error: ||lambda^(k) - lambda^(k-1)||_2
        erk = float(np.linalg.norm(lambda_k - lambda_prev, ord=2))

        # Actualiza B para la siguiente iteración
        B = C

        # Criterio de parada
        if erk < tol:
            conv = 1
            break

        lambda_prev = lambda_k

    # Reordena las columnas de V para alinearlas con lambdas y normalizar
    V_sorted = V[:, idx_orden]

    # Asegura la normalización de los vectores propios (||v_j||_2 = 1)
    for j in range(n):
        norma_v = np.linalg.norm(V_sorted[:, j], ord=2)
        if norma_v > 0:
            V_sorted[:, j] /= norma_v

    # Formatea el vector de valores propios como matriz columna (n x 1)
    lambdas = lambda_k.reshape(-1, 1)

    return lambdas, V_sorted, erk, k, conv

def metodo_Jacobi(A, iterMax, tol):
    """
    Método de Jacobi para la aproximación de valores y vectores 
    propios de una matriz simétrica semidefinida positiva. 

    Parámetros:
    -----------
    A       : numpy.ndarray - Matriz cuadrada simétrica semidefinida positiva (n x n). 
    iterMax : int           - Número máximo de iteraciones permitidas. 
    tol     : float         - Tolerancia para el criterio de parada.

    Retorna:
    --------
    lambdas : numpy.ndarray - Vector columna (n x 1) con los valores propios ordenados de mayor a menor. 
    V_sorted: numpy.ndarray - Matriz (n x n) con los vectores propios normalizados como columnas. 
    erk     : float         - Error de la última iteración ||lambda^(k) - lambda^(k-1)||_2. 
    k       : int           - Número de iteraciones ejecutadas. 
    conv    : int           - 1 si el método convergió, 0 en caso contrario.
    """
    n = A.shape[0]

    # Inicialización 
    B = np.copy(A).astype(float)
    V = np.eye(n, dtype=float)
    k = 0
    conv = 0
    erk = float("inf")

    # Extrae la diagonal de A y la ordena de mayor a menor
    diag_B = np.diag(B)
    idx_orden = np.argsort(diag_B)[::-1]
    lambda_prev = diag_B[idx_orden]

    # Bucle principal 
    while k < iterMax:
        # Busca el elemento máximo fuera de la diagonal principal
        off_diag = np.abs(B).copy()
        np.fill_diagonal(off_diag, 0.0)
        p, q = np.unravel_index(np.argmax(off_diag), off_diag.shape)

        # Cálculo del ángulo theta
        if B[p, p] == B[q, q]:
            theta = np.pi / 4.0
        else:
            theta = 0.5 * np.arctan((2.0 * B[p, q]) / (B[p, p] - B[q, q]))

        # Seno y coseno de la rotación
        c = np.cos(theta)
        s = np.sin(theta)

        #  Construcción de la matriz de rotación P
        P = np.eye(n, dtype=float)
        P[p, p] = c
        P[q, q] = c
        P[p, q] = -s
        P[q, p] = s

        # Transformación de similitud y acumulación de vectores propios
        B = P.T @ B @ P
        V = V @ P

        # Extrae y ordena lambda^(k) de mayor a menor
        diag_B = np.diag(B)
        idx_orden = np.argsort(diag_B)[::-1]
        lambda_k = diag_B[idx_orden]

        # Cálculo del error: ||lambda^(k) - lambda^(prev)||_2
        erk = float(np.linalg.norm(lambda_k - lambda_prev, ord=2))

        # Incrementa la iteración
        k += 1

        # Criterio de parada
        if erk < tol:
            conv = 1
            break

        # Actualizar lambda_prev
        lambda_prev = lambda_k

    # Reordena las columnas de V usando el orden descendente de lambda^(k)
    V_sorted = V[:, idx_orden]

    # Normaliza cada columna j de V para garantizar ||v_j||_2 = 1
    for j in range(n):
        norma_v = np.linalg.norm(V_sorted[:, j], ord=2)
        if norma_v > 0:
            V_sorted[:, j] /= norma_v

    # Da formato de vector columna (n x 1)
    lambdas = lambda_k.reshape(-1, 1)

    # Retorna resultados
    return lambdas, V_sorted, erk, k, conv