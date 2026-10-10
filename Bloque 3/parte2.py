# =========================================================================
# INSTITUTO TECNOLÓGICO DE COSTA RICA
# CE1111: Análisis Numérico para Ingeniería
# Escuela de Matemática
#
# Portafolio Computacional - Bloque 3 (Parte 2)
# Autores: Joaquin Ignacio Ramírez Sequeira y Joseph Stif Piedra Montero
# =========================================================================

import numpy as np
import time
import matplotlib.pyplot as plt
from parte1 import metodo_QR, metodo_Jacobi

n = 200
iterMax = 10000
tol = 1e-8

# Construcción de la matriz A tridiagonal simétrica definida positiva (200x200)
A = np.diag(4 * np.ones(n)) + np.diag(-1 * np.ones(n - 1), 1) + np.diag(-1 * np.ones(n - 1), -1)

# Ejecución y medición de tiempo de los métodos
print("Ejecutando Método QR...")
inicio_qr = time.time()
lambda_qr, V_qr, err_qr, k_qr, conv_qr = metodo_QR(A, iterMax, tol)
fin_qr = time.time()
tiempo_qr = fin_qr - inicio_qr

print("Ejecutando Método de Jacobi...")
inicio_jacobi = time.time()
lambda_jacobi, V_jacobi, err_jacobi, k_jacobi, conv_jacobi = metodo_Jacobi(A, iterMax, tol)
fin_jacobi = time.time()
tiempo_jacobi = fin_jacobi - inicio_jacobi

# Comprobaciones numéricas (residuos y ortogonalidad)
def calcular_metricas(A, lambdas, V):
    n = A.shape[0]
    residuos = np.zeros(n)
    
    # Comprobación de pares propios r_j = ||A * v_j - lambda_j * v_j||_2
    for j in range(n):
        v_j = V[:, j].reshape(-1, 1)
        lambda_j = lambdas[j, 0]
        residuos[j] = np.linalg.norm(A @ v_j - lambda_j * v_j, ord=2)
        
    r_max = np.max(residuos)
    
    # Error de ortogonalidad ||V^T * V - I||_F
    I = np.eye(n)
    err_ort = np.linalg.norm(V.T @ V - I, ord='fro')
    
    return r_max, err_ort

r_max_qr, err_ort_qr = calcular_metricas(A, lambda_qr, V_qr)
r_max_jacobi, err_ort_jacobi = calcular_metricas(A, lambda_jacobi, V_jacobi)

# Despliegue de resultados en terminal
print("\n" + "="*90)
print("VECTORES Y VALORES PROPIOS OBTENIDOS (MUESTRA DE LOS PRIMEROS 5)")
print("="*90)
print(f"{'MÉTODO QR':<45} | {'MÉTODO DE JACOBI':<45}")
print("-" * 90)
for i in range(5):
    print(f"L_{i+1}: {lambda_qr[i, 0]:.6f} -> v_{i+1}: {V_qr[:3, i]}... | L_{i+1}: {lambda_jacobi[i, 0]:.6f} -> v_{i+1}: {V_jacobi[:3, i]}...")

print("\n" + "="*115)
print("TABLA COMPARATIVA DE RESULTADOS - BLOQUE 3 PARTE 2")
print("="*115)
print(f"{'Método':<15} | {'Error final (erk)':<18} | {'Iteraciones (k)':<15} | {'Tiempo (s)':<12} | {'Conv':<5} | {'Residuo max (r_max)':<20} | {'Err Ortogonalidad':<20}")
print("-" * 115)
print(f"{'QR':<15} | {err_qr:<18.4e} | {k_qr:<15} | {tiempo_qr:<12.4f} | {conv_qr:<5} | {r_max_qr:<20.4e} | {err_ort_qr:<20.4e}")
print(f"{'Jacobi':<15} | {err_jacobi:<18.4e} | {k_jacobi:<15} | {tiempo_jacobi:<12.4f} | {conv_jacobi:<5} | {r_max_jacobi:<20.4e} | {err_ort_jacobi:<20.4e}")
print("="*115 + "\n")

# Análisis Comparativo
print("="*90)
print("ANÁLISIS COMPARATIVO DE LOS RESULTADOS")
print("="*90)
analisis = """
1. APROXIMACIÓN DE VALORES PROPIOS:
Los métodos presentan discrepancias desde el tercer decimal en sus aproximaciones 
(ej. L_1 QR ~ 5.999 vs L_1 Jacobi ~ 5.993). Esto evidencia que ninguno de los dos 
métodos logró converger hacia los valores propios exactos dentro del límite permitido.

2. COMPORTAMIENTO DE LOS RESIDUOS (||A*v_j - lambda_j*v_j||_2):
Al no existir convergencia, los residuos máximos (r_max) son significativos: 
~8.58e-04 para QR y ~4.94e-02 para Jacobi. La igualdad fundamental de pares propios 
no se cumple en exactitud porque las iteraciones fueron interrumpidas prematuramente.

3. NORMALIZACIÓN Y ORTOGONALIDAD DE LOS VECTORES PROPIOS:
A pesar de la falta de convergencia de los valores propios, la ortogonalidad sí se 
cumple (errores del orden de 10^-13 y 10^-14). Esto es teóricamente 
correcto, ya que tanto la factorización QR como las rotaciones de Jacobi son 
transformaciones estrictamente ortogonales que preservan esta propiedad en cada iteración.

4. NÚMERO DE ITERACIONES Y TIEMPO DE EJECUCIÓN:
Ambos métodos registraron exactamente 10000 iteraciones, chocando contra el límite iterMax. 
Sin embargo, el tiempo de ejecución marca una diferencia: QR demoró ~75.8s 
mientras que Jacobi tomó solo ~10.3s. Esto ocurre porque una iteración de QR implica 
una costosa descomposición matricial completa, mientras que Jacobi solo realiza 
actualizaciones locales mediante una única rotación plana por iteración.

5. DIFERENCIAS EN EL CRITERIO DE PARADA Y CONVERGENCIA:
Ningún método alcanzó la tolerancia requerida (10^-8), resultando en una convergencia 
igual a 0. Para el caso de Jacobi, 10000 iteraciones en una matriz de 200x200 no alcanzó 
ni para completar un barrido entero (n*(n-1)/2 = 19900 elementos). El método QR básico, 
sin técnicas de desplazamiento (shifts), muestra una convergencia bastante lenta para matrices 
grandes.
"""
print(analisis)
print("="*90)

# Generación de gráficas
metodos = ['Método QR', 'Método Jacobi']
errores = [err_qr, err_jacobi]
tiempos = [tiempo_qr, tiempo_jacobi]
iteraciones = [k_qr, k_jacobi]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Comparación de Métodos de Valores Propios - Matriz Tridiagonal (200x200)', fontsize=14, fontweight='bold')

# Gráfica 1: Error final (escala logarítmica)
ax1.bar(metodos, errores, color=['#1f77b4', '#ff7f0e'])
ax1.set_yscale('log')
ax1.set_title('Error final (erk)')
ax1.set_ylabel('Error ||λ^(k) - λ^(k-1)||_2')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Gráfica 2: Tiempo de ejecución
ax2.bar(metodos, tiempos, color=['#2ca02c', '#d62728'])
ax2.set_title('Tiempo de Ejecución')
ax2.set_ylabel('Tiempo (segundos)')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Gráfica 3: Número de iteraciones (escala logarítmica)
ax3.bar(metodos, iteraciones, color=['#9467bd', '#8c564b'])
ax3.set_yscale('log')
ax3.set_title('Número de Iteraciones')
ax3.set_ylabel('Iteraciones (escala log)')
ax3.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()