"""
INSTITUTO TECNOLÓGICO DE COSTA RICA
CE-1111: Análisis Numérico para Ingeniería
Semestre II-2026
Proyecto de Investigación - Simulación Térmica Transitoria de un Microprocesador

Autores: Joseph Stif Piedra Montero
         Joaquin Ignacio Ramírez Sequeira

Script principal: simulacion_termica.py
Descripción:
    Resuelve el modelo térmico transitorio no lineal de un encapsulado de
    microprocesador con disipación de potencia variable, convección y radiación
    térmica (Ley de Stefan-Boltzmann):
        m * cp * dT/dt = P(t) - h_conv * A * (T - T_inf) - eps * sigma * A * (T^4 - T_inf^4)
    utilizando los métodos numéricos implementados en metodos_edo.py:
        - Euler explícito
        - Heun (RK2)
        - Runge-Kutta de cuarto orden (RK4)

Salidas generadas:
    1. Reporte detallado en terminal con: nombre del método, valores iniciales,
       número de pasos/iteraciones, aproximación numérica, error y tiempo (s).
    2. Estudio de convergencia al refinar el paso temporal h.
    3. Gráficas guardadas en disco:
       - evolucion_temperatura.png
       - convergencia_errores.png
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from metodos_edo import euler_explicito, heun_rk2, rk4_clasico


# =============================================================================
# 1. DEFINICIÓN DE PARÁMETROS FÍSICOS DEL MICROPROCESADOR (SISTEMA INTERNACIONAL)
# =============================================================================
MASA = 0.045                 # Masa del conjunto encapsulado-disipador [kg]
CP = 900.0                   # Calor específico del aluminio [J/(kg*K)]
CAP_TERMICA = MASA * CP      # Capacitancia térmica total C_th = m * cp [J/K]

AREA = 0.012                 # Área superficial efectiva de transferencia [m^2]
H_CONV = 18.0                # Coeficiente de transferencia por convección [W/(m^2*K)]
EMISIVIDAD = 0.85            # Emisividad superficial del disipador anodizado [-]
SIGMA = 5.670374419e-8       # Constante de Stefan-Boltzmann [W/(m^2*K^4)]

T_INF = 298.15               # Temperatura ambiente (25.0 °C) [K]
T0 = 313.15                  # Temperatura inicial del procesador (40.0 °C) [K]

# Parámetros de la disipación dinámica de potencia P(t) = P0 + P1 * sin(omega * t)
P0 = 18.0                    # Potencia promedio disipada [W]
P1 = 10.0                    # Amplitud de oscilación por ráfagas de carga [W]
PERIODO_CARGA = 30.0         # Período del ciclo de trabajo [s]
OMEGA = 2.0 * np.pi / PERIODO_CARGA  # Frecuencia angular [rad/s]

# Intervalo temporal de simulación [t0, tf]
T_INICIAL = 0.0              # Tiempo inicial [s]
T_FINAL = 180.0              # Tiempo final (3 minutos) [s]


def potencia_disipada(t):
    """Calcula la potencia térmica instantánea P(t) en Watts."""
    return P0 + P1 * np.sin(OMEGA * t)


def edo_termica(t, T):
    """
    Define el lado derecho f(t, T) = dT/dt de la EDO térmica no lineal.

    Parámetros:
        t (float) : Tiempo actual [s].
        T (float) : Temperatura instantánea del componente [K].

    Retorna:
        dT_dt (float) : Tasa de cambio temporal de la temperatura [K/s].
    """
    q_gen = potencia_disipada(t)
    q_conv = H_CONV * AREA * (T - T_INF)
    q_rad = EMISIVIDAD * SIGMA * AREA * (T**4 - T_INF**4)
    return (q_gen - q_conv - q_rad) / CAP_TERMICA


def calcular_solucion_referencia(n_ref=180000):
    """
    Genera una trayectoria de referencia de alta precisión mediante RK4 clásico
    con paso temporal ultra-fino (h_ref = 0.001 s), dado que la EDO no lineal
    carece de solución analítica cerrada en funciones elementales.
    """
    t_ref, T_ref = rk4_clasico(edo_termica, T_INICIAL, T_FINAL, T0, n_ref)
    return t_ref, T_ref


def ejecutar_experimento_principal(n_pasos=180):
    """
    Ejecuta la comparación de los tres métodos numéricos para un tamaño de paso
    fijo h = (T_FINAL - T_INICIAL) / n_pasos y despliega en consola todas las
    métricas exigidas.
    """
    h = (T_FINAL - T_INICIAL) / n_pasos

    # Cálculo de la solución de referencia para estimar el error numérico
    t_ref, T_ref = calcular_solucion_referencia(n_ref=180000)

    metodos = [
        ("Euler Explícito (Orden 1)", euler_explicito),
        ("Heun / Predictor-Corrector (RK2, Orden 2)", heun_rk2),
        ("Runge-Kutta Clásico (RK4, Orden 4)", rk4_clasico),
    ]

    resultados = {}

    print("=" * 88)
    print("SIMULACIÓN TÉRMICA TRANSITORIA DE MICROPROCESADOR (EDO NO LINEAL)")
    print("=" * 88)
    print(f"Modelo: m*cp*(dT/dt) = P(t) - h_conv*A*(T - T_inf) - eps*sigma*A*(T^4 - T_inf^4)")
    print(f"Intervalo temporal : [{T_INICIAL:.1f}, {T_FINAL:.1f}] s")
    print(f"Condición inicial  : T(0) = {T0:.2f} K ({T0 - 273.15:.2f} °C)")
    print(f"Temperatura amb.   : T_inf = {T_INF:.2f} K ({T_INF - 273.15:.2f} °C)")
    print(f"Discretización     : n = {n_pasos} pasos | Tamaño de paso h = {h:.4f} s")
    print("=" * 88)

    for nombre, funcion_metodo in metodos:
        # Medición precisa del tiempo de ejecución en segundos
        t_inicio = time.perf_counter()
        t_num, T_num = funcion_metodo(edo_termica, T_INICIAL, T_FINAL, T0, n_pasos)
        t_fin = time.perf_counter()
        tiempo_ejecucion = t_fin - t_inicio

        # Interpolación/muestreo exacto de la referencia sobre los nodos t_num
        # Como n_ref = 180000 es múltiplo exacto de n_pasos, se extraen los índices exactos
        factor = len(t_ref) - 1
        indices_ref = np.linspace(0, factor, n_pasos + 1, dtype=int)
        T_ref_nodos = T_ref[indices_ref]

        # Cálculo de errores absoluto máximo global y en el instante final
        errores_puntuales = np.abs(T_num - T_ref_nodos)
        error_max_global = np.max(errores_puntuales)
        error_final = errores_puntuales[-1]

        T_final_K = T_num[-1]
        T_final_C = T_final_K - 273.15
        T_max_C = np.max(T_num) - 273.15

        resultados[nombre] = {
            "t": t_num,
            "T": T_num,
            "error_max": error_max_global,
            "error_final": error_final,
            "tiempo": tiempo_ejecucion,
            "n": n_pasos,
            "h": h,
        }

        # Despliegue estructurado en consola 
        print(f"\n>>> MÉTODO: {nombre}")
        print(f"    Valores iniciales          : t0 = {T_INICIAL:.2f} s, T0 = {T0:.2f} K ({T0 - 273.15:.2f} °C), h = {h:.4f} s")
        print(f"    Número de iteraciones (n)  : {n_pasos}")
        print(f"    Aproximación en t = {T_FINAL:.0f} s  : T({T_FINAL:.0f}) = {T_final_K:.6f} K ({T_final_C:.4f} °C)")
        print(f"    Temperatura pico alcanzada : T_max = {T_max_C:.4f} °C")
        print(f"    Error máximo global (L_inf): {error_max_global:.6e} K")
        print(f"    Error absoluto en t_final  : {error_final:.6e} K")
        print(f"    Tiempo de ejecución        : {tiempo_ejecucion:.6f} segundos")

    print("\n" + "=" * 88)
    print("MUESTRA DE EVOLUCIÓN TEMPORAL DE TEMPERATURA (°C) EN INSTANTES CLAVE")
    print("=" * 88)
    print(f"{'Tiempo (s)':<12} | {'Euler (°C)':<14} | {'Heun RK2 (°C)':<15} | {'RK4 (°C)':<14} | {'Referencia (°C)':<15}")
    print("-" * 88)

    tiempos_muestra = [0, 30, 60, 90, 120, 150, 180]
    for tm in tiempos_muestra:
        idx = int(round(tm / h))
        idx_ref = int(round(tm / ((T_FINAL - T_INICIAL) / 180000)))
        t_eu = resultados["Euler Explícito (Orden 1)"]["T"][idx] - 273.15
        t_he = resultados["Heun / Predictor-Corrector (RK2, Orden 2)"]["T"][idx] - 273.15
        t_rk = resultados["Runge-Kutta Clásico (RK4, Orden 4)"]["T"][idx] - 273.15
        t_rf = T_ref[idx_ref] - 273.15
        print(f"{tm:<12d} | {t_eu:<14.4f} | {t_he:<15.4f} | {t_rk:<14.4f} | {t_rf:<15.4f}")

    return resultados, t_ref, T_ref


def ejecutar_estudio_convergencia(t_ref, T_ref):
    """
    Evalúa la reducción del error global máximo E_inf al disminuir el tamaño
    de paso h (refinamiento de malla) para verificar experimentalmente el orden
    de convergencia teórico O(h^p) de cada método.
    """
    lista_n = [45, 90, 180, 360, 720, 1440]
    hs = []
    err_euler = []
    err_heun = []
    err_rk4 = []

    print("\n" + "=" * 88)
    print("ESTUDIO DE CONVERGENCIA AL REFINAR EL PASO TEMPORAL h")
    print("=" * 88)
    print(f"{'n pasos':<10} | {'Paso h (s)':<12} | {'Error Euler (K)':<18} | {'Error Heun (K)':<18} | {'Error RK4 (K)':<18}")
    print("-" * 88)

    for n in lista_n:
        h = (T_FINAL - T_INICIAL) / n
        hs.append(h)

        indices_ref = np.linspace(0, len(t_ref) - 1, n + 1, dtype=int)
        T_ref_nodos = T_ref[indices_ref]

        _, T_eu = euler_explicito(edo_termica, T_INICIAL, T_FINAL, T0, n)
        _, T_he = heun_rk2(edo_termica, T_INICIAL, T_FINAL, T0, n)
        _, T_rk = rk4_clasico(edo_termica, T_INICIAL, T_FINAL, T0, n)

        e_eu = np.max(np.abs(T_eu - T_ref_nodos))
        e_he = np.max(np.abs(T_he - T_ref_nodos))
        e_rk = np.max(np.abs(T_rk - T_ref_nodos))

        err_euler.append(e_eu)
        err_heun.append(e_he)
        err_rk4.append(e_rk)

        print(f"{n:<10d} | {h:<12.4f} | {e_eu:<18.6e} | {e_he:<18.6e} | {e_rk:<18.6e}")

    print("=" * 88)
    return np.array(hs), np.array(err_euler), np.array(err_heun), np.array(err_rk4)


def generar_graficas(resultados, t_ref, T_ref, hs, err_euler, err_heun, err_rk4):
    """
    Genera y guarda las figuras comparativas en alta resolución para el informe
    y la presentación académica.
    """
    # --- FIGURA 1: Evolución temporal de la temperatura ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True,
                                     gridspec_kw={"height_ratios": [2.2, 1]})

    ax1.plot(t_ref, T_ref - 273.15, "k-", linewidth=2.0, label="Referencia (RK4, h = 0.001 s)")
    ax1.plot(resultados["Euler Explícito (Orden 1)"]["t"],
             resultados["Euler Explícito (Orden 1)"]["T"] - 273.15,
             "o--", markersize=3.5, alpha=0.8, label="Euler Explícito (h = 1.0 s)")
    ax1.plot(resultados["Heun / Predictor-Corrector (RK2, Orden 2)"]["t"],
             resultados["Heun / Predictor-Corrector (RK2, Orden 2)"]["T"] - 273.15,
             "s:", markersize=3.5, alpha=0.85, label="Heun RK2 (h = 1.0 s)")
    ax1.plot(resultados["Runge-Kutta Clásico (RK4, Orden 4)"]["t"],
             resultados["Runge-Kutta Clásico (RK4, Orden 4)"]["T"] - 273.15,
             "^", markersize=3.5, alpha=0.85, label="RK4 Clásico (h = 1.0 s)")

    ax1.set_ylabel("Temperatura del Procesador (°C)", fontsize=11)
    ax1.set_title("Evolución Térmica Transitoria con Convección, Radiación y Carga Dinámica",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=9.5)

    # Subgráfica inferior: Perfil de potencia disipada P(t)
    ax2.plot(t_ref, potencia_disipada(t_ref), color="#d62728", linewidth=1.5)
    ax2.set_xlabel("Tiempo t (s)", fontsize=11)
    ax2.set_ylabel("Potencia P(t) (W)", fontsize=11)
    ax2.set_title("Perfil de Disipación de Potencia Dinámica P(t) = 18 + 10·sin(2πt/30)",
                  fontsize=10.5)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("evolucion_temperatura.png", dpi=300)

    # --- FIGURA 2: Convergencia del error global en escala log-log ---
    fig2, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.loglog(hs, err_euler, "o-", linewidth=2, markersize=6, label="Euler Explícito — Orden teórico O(h)")
    ax.loglog(hs, err_heun, "s-", linewidth=2, markersize=6, label="Heun (RK2) — Orden teórico O(h²)")
    ax.loglog(hs, err_rk4, "^-", linewidth=2, markersize=6, label="RK4 Clásico — Orden teórico O(h⁴)")

    ax.set_xlabel("Tamaño de paso temporal h (s)", fontsize=11)
    ax.set_ylabel("Error máximo global ||T_num - T_ref||_∞ (K)", fontsize=11)
    ax.set_title("Análisis de Convergencia de los Métodos Numéricos (Escala Log-Log)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("convergencia_errores.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Ejecuta experimento principal con n = 180 (h = 1.0 s)
    resultados_exp, t_referencia, T_referencia = ejecutar_experimento_principal(n_pasos=180)

    # Ejecuta análisis de refinamiento de paso h
    hs_arr, e_eu_arr, e_he_arr, e_rk_arr = ejecutar_estudio_convergencia(t_referencia, T_referencia)

    # Genera y guarda gráficas de resultados
    generar_graficas(resultados_exp, t_referencia, T_referencia,
                     hs_arr, e_eu_arr, e_he_arr, e_rk_arr)