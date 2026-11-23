# Proyecto de Investigación — CE-1111: Análisis Numérico para Ingeniería
## Modelado de la Evolución Térmica Transitoria de un Microprocesador mediante una EDO No Lineal

### 1. Descripción de los archivos
- `metodos_edo.py`: Implementación computacional desde cero de los métodos numéricos de **Euler explícito**, **Heun (RK2)** y **Runge-Kutta clásico de cuarto orden (RK4)** para problemas de valor inicial (Cauchy).
- `simulacion_termica.py`: Script principal que resuelve el balance térmico transitorio con convección, radiación superficial ($T^4$) y potencia dinámica $P(t)$, imprime las métricas de cada método en consola y genera las gráficas comparativas.
- `requirements.txt`: Lista de bibliotecas externas requeridas (`numpy` y `matplotlib`).

### 2. Requisitos y preparación del entorno
Se requiere **Python 3.9 o superior**. Para ejecutar en un entorno limpio:

```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt