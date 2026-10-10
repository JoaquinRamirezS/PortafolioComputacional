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
    Método QR para la aproximación de los valores y vectores propios de 
    una matriz simétrica semidefinida positiva.
    
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

    # 1. Inicialización 
    B = np.copy(A).astype(float)
    V = np.eye(n, dtype=float)
    conv = 0

    # Vector lambda inicial ordenado de mayor a menor
    lambda_prev = np.sort(np.diag(B))[::-1]
    erk = float("inf")
    idx_orden = np.arange(n)

    # 2. Bucle iterativo
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

        # 3. Extracción y ordenamiento decreciente de valores propios
        diag_C = np.diag(C)
        idx_orden = np.argsort(diag_C)[::-1]
        lambda_k = diag_C[idx_orden]

        # 4. Cálculo del error: ||lambda^(k) - lambda^(k-1)||_2
        erk = float(np.linalg.norm(lambda_k - lambda_prev, ord=2))

        # Actualiza B para la siguiente iteración
        B = C

        # 5. Criterio de parada
        if erk < tol:
            conv = 1
            break

        lambda_prev = lambda_k

    # 6. Reordena las columnas de V para alinearlas con lambdas y normalizar
    V_sorted = V[:, idx_orden]

    # Asegura la normalización de los vectores propios (||v_j||_2 = 1)
    for j in range(n):
        norma_v = np.linalg.norm(V_sorted[:, j], ord=2)
        if norma_v > 0:
            V_sorted[:, j] /= norma_v

    # Formatea el vector de valores propios como matriz columna (n x 1)
    lambdas = lambda_k.reshape(-1, 1)

    return lambdas, V_sorted, erk, k, conv