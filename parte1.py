##############################################################################
# CE1111: Análisis Numérico para Ingeniería
# Escuela de Ingeniería en Computadores
#
# Portafolio Bloque 1:Parte 1
#
# Autores: Joaquin Ignacio Ramírez Sequeira
#          Joseph Stif Piedra Montero 
# Este archivo implementa computacionalmente los siguientes métodos para la 
# aproximación de soluciones de ecuaciones no lineales:Newton-Raphson,Secante,
# Steffensen, Müller, Bisección, Falsa Posición.
##############################################################################
import sympy as sp

##############################################################################
# Método 1:Newton-Raphson
##############################################################################

def Newton_Raphson(f,x0,tol,maxIter):
    """
    Esta función permite aproximar la solución de ecuaciones no lineales 
    aplicando el método de Newton-Raphson.

    Parámetros
    ----------
    f : string 
        Función ingresada en formato texto.
    x0 : float
        Valor inicial
    tol: float
        Tolerancia
    maxIter: integer
        Número máximo de iteraciones

    Retorna
    -------
    xK : float
        Aproximación obtenida
    er_k : float
        Error correspondiente a la última iteración realizada
    k : integer
        Número de iteraciones ejecutadas
    conv : integer
        Variable que indique si el método alcanzó la tolerancia solicitada antes de
        llegar al número máximo de iteraciones
    """
    # Definir x como simbólica
    x = sp.Symbol('x')

    # Conversión de función:Texto a simbólico
    f_sym=sp.sympify(f)

    # Primer derivada de f_sym
    f_der1=sp.diff(f_sym,x)

    # Definición de xk con dato de valor inicial
    xk=x0

    # Inicialización del conteo de las iteraciones
    k=0

    # Evaluar f_sym con xk
    f_xk = f_sym.subs(x,xk).evalf()

    # Evaluar f_der1 con xk
    f_der_xk=f_der1.subs(x,xk).evalf()

    # Error
    er_k=abs(f_xk)

    # Condición de iteración
    while k<maxIter and er_k>tol:
        if f_der_xk == 0:
            print("La derivada de f alcanzó el valor de 0.No es posible iterar más")
            break
        k += 1 # Aumento de k
        xk = xk - (f_xk/f_der_xk) #Actualización de la aproximación
        f_xk = f_sym.subs(x,xk).evalf() # Actualización de f_xk por iteración
        f_der_xk= f_der1.subs(x,xk).evalf() # Actualización de f_der1 por iteración
        er_k = abs(f_xk) # Actualización de error

    # Condición de convergencia
    if er_k <= tol:
       conv =  1
    else:
        conv = 0

    return xk,er_k,k,conv

##############################################################################
# Método 2:Secante
##############################################################################

def Secante(f,x0,x1,tol,maxIter):
    """
    Esta función permite aproximar la solución de ecuaciones no lineales 
    aplicando el método de la Secante.

    Parámetros
    ----------
    f : string 
        Función ingresada en formato texto.
    x0 : float
        Primer valor inicial
    x1 : float
        Segundo valor inicial
    tol: float
        Tolerancia
    maxIter: integer
        Número máximo de iteraciones

    Retorna
    -------
    xk : float
        Aproximación obtenida
    er_k : float
        Error correspondiente a la última iteración realizada
    k : integer
        Número de iteraciones ejecutadas
    conv : integer
        Variable que indique si el método alcanzó la tolerancia solicitada antes de
        llegar al número máximo de iteraciones
    """
    # Definir x como simbólica
    x = sp.Symbol('x')

    # Conversión de función:Texto a simbólico
    f_sym =sp.sympify(f)

    # Inicialización del conteo de las iteraciones
    k=0
    # Evaluar f_sym con los valores iniciales
    f_xk = f_sym.subs(x,x0).evalf() #Con valor inicial x0
    f_xk_1 = f_sym.subs(x,x1).evalf() #Con valor inicial x1

    #Error
    xk = x1
    er_k = abs(f_xk_1)

    # Condición de iteración
    while k<maxIter and er_k>tol:
        if (f_xk_1-f_xk) == 0:
            print("El denominador se hizo 0.No se puede continuar")
            break
        k += 1 # Aumento de k
        xk = x1-f_xk_1*((x1-x0)/(f_xk_1-f_xk)) #Actualización de la aproximación
        x0,x1=x1,xk #Actualizacion de valores iniciales
        f_xk = f_xk_1# Actualización de f_xk por iteración
        f_xk_1 = f_sym.subs(x,xk).evalf() # Actualización de f_xk_1 por iteración
        er_k = abs(f_xk_1) # Actualización de error

    # Condición de convergencia
    if er_k <= tol:
       conv =  1
    else:
        conv = 0
    return xk,er_k,k,conv

##############################################################################
# Método 3:Steffensen
##############################################################################

def Steffensen(f,x0,tol,maxIter):
    """
    Esta función permite aproximar la solución de ecuaciones no lineales 
    aplicando el método de Steffensen.

    Parámetros
    ----------
    f : string 
        Función ingresada en formato texto.
    x0 : float
        Valor inicial
    tol: float
        Tolerancia
    maxIter: integer
        Número máximo de iteraciones

    Retorna
    -------
    xK : float
        Aproximación obtenida
    er_k : float
        Error correspondiente a la última iteración realizada
    k : integer
        Número de iteraciones ejecutadas
    conv : integer
        Variable que indique si el método alcanzó la tolerancia solicitada antes de
        llegar al número máximo de iteraciones
    """
    # Definir x como simbólica
    x = sp.Symbol('x')

    # Conversión de función:Texto a simbólico
    f_sym=sp.sympify(f)

    # Definición de xk con dato de valor inicial
    xk=x0

    # Inicialización del conteo de las iteraciones
    k=0

    # Evaluar f_sym con xk y con xk+f_xk
    f_xk = f_sym.subs(x,xk).evalf()
    f_xk1= f_sym.subs(x,(xk+f_xk)).evalf()

    # Error
    er_k=abs(f_xk)

    # Condición de iteración
    while k<maxIter and er_k>tol:
        if  f_xk1 - f_xk == 0:
            print("El denominador se hizo 0.No se puede continuar")
            break
        k += 1 # Aumento de k
        xk = xk - ((f_xk)**2 /(f_xk1-f_xk)) #Actualización de la aproximación
        f_xk = f_sym.subs(x,xk).evalf() # Actualización de f_xk por iteración
        f_xk1 = f_sym.subs(x,(xk+f_xk)).evalf() # Actualización de f_xk1 por iteración
        er_k = abs(f_xk) # Actualización de error

    # Condición de convergencia
    if er_k <= tol:
       conv =  1
    else:
        conv = 0

    return xk,er_k,k,conv

##############################################################################
# Método 4: Müller
##############################################################################

def Muller(f, x0, x1, x2, tol, maxIter):
    """
    Esta función permite aproximar la solución de ecuaciones no lineales 
    aplicando el método de Müller estrictamente como se especifica en el documento.

    Parámetros
    ----------
    f : string 
        Función ingresada en formato texto.
    x0 : float
        Primer valor inicial
    x1 : float
        Segundo valor inicial
    x2 : float
        Tercer valor inicial
    tol: float
        Tolerancia
    maxIter: integer
        Número máximo de iteraciones

    Retorna
    -------
    xk : float
        Aproximación obtenida
    er_k : float
        Error correspondiente a la última iteración realizada
    k : integer
        Número de iteraciones ejecutadas
    conv : integer
        Variable que indica si el método alcanzó la tolerancia solicitada
    """
    # Definir x como simbólica
    x = sp.Symbol('x')

    # Conversión de función: Texto a simbólico
    f_sym = sp.sympify(f)

    # Inicialización del conteo de las iteraciones
    k = 0

    # Evaluar f_sym con los tres valores iniciales
    f_x0 = f_sym.subs(x, x0).evalf()
    f_x1 = f_sym.subs(x, x1).evalf()
    f_x2 = f_sym.subs(x, x2).evalf()

    # Definición inicial de xk y error
    xk = x2
    er_k = abs(f_x2)

    # Condición de iteración
    while k < maxIter and er_k > tol:
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if denom == 0:
            print("El denominador se hizo 0. No se puede continuar.")
            break

        # Coeficientes a, b y c del polinomio p(x) = a(x-x2)^2 + b(x-x2) + c
        c = f_x2
        b = ((x0 - x2)**2 * (f_x1 - f_x2) - (x1 - x2)**2 * (f_x0 - f_x2)) / denom
        a = ((x1 - x2) * (f_x0 - f_x2) - (x0 - x2) * (f_x1 - f_x2)) / denom

        disc = b**2 - 4 * a * c
        
        # Signo de b
        sgn_b = 1 if float(sp.re(b)) >= 0 else -1

        den_rad = b + sgn_b * sp.sqrt(disc)
        if den_rad == 0:
            print("El denominador del radical se hizo 0. No se puede continuar.")
            break

        k += 1 # Aumento de k
        
        # Actualización de la aproximación r usando la fórmula cuadrática racionalizada
        xk = x2 - (2 * c) / den_rad
        xk = xk.evalf()

        # Selección de los dos puntos más cercanos a xk de entre {x0, x1, x2}
        puntos = [(x0, f_x0), (x1, f_x1), (x2, f_x2)]
        puntos_ordenados = sorted(puntos, key=lambda p: abs(p[0] - xk))

        # Asignación de los dos más cercanos como x0 y x1, y xk como el nuevo x2
        (x0, f_x0), (x1, f_x1) = puntos_ordenados[0], puntos_ordenados[1]
        x2 = xk
        f_x2 = f_sym.subs(x, x2).evalf()
        
        er_k = abs(f_x2) # Actualización del error

    # Condición de convergencia
    if er_k <= tol:
        conv = 1
    else:
        conv = 0

    return xk, er_k, k, conv

##############################################################################
# Método 5: Bisección
##############################################################################

def Biseccion(f, a, b, tol, maxIter):
    """
    Esta función permite aproximar la solución de ecuaciones no lineales 
    aplicando el método de la Bisección.

    Parámetros
    ----------
    f : string 
        Función ingresada en formato texto.
    a : float
        Límite inferior del intervalo [a, b]
    b : float
        Límite superior del intervalo [a, b]
    tol: float
        Tolerancia
    maxIter: integer
        Número máximo de iteraciones

    Retorna
    -------
    xk : float
        Aproximación obtenida
    er_k : float
        Error correspondiente a la última iteración realizada
    k : integer
        Número de iteraciones ejecutadas
    conv : integer
        Variable que indique si el método alcanzó la tolerancia solicitada antes de
        llegar al número máximo de iteraciones
    """
    # Definir x como simbólica
    x = sp.Symbol('x')

    # Conversión de función: Texto a simbólico
    f_sym = sp.sympify(f)

    # Evaluar la función en los extremos del intervalo
    f_a = f_sym.subs(x, a).evalf()
    f_b = f_sym.subs(x, b).evalf()

    # Verificar si alguna de las fronteras ya es raíz
    if f_a == 0:
        return float(a), 0.0, 0, 1
    if f_b == 0:
        return float(b), 0.0, 0, 1

    # Verificar condición de cambio de signo (Teorema de Bolzano)
    if f_a * f_b > 0:
        print("El intervalo [a, b] no cumple la condición de cambio de signo f(a)*f(b) < 0.")
        return None, None, 0, 0

    # Inicialización del conteo de las iteraciones
    k = 0
    xk = (a + b) / 2.0
    f_xk = f_sym.subs(x, xk).evalf()
    er_k = abs(f_xk)

    # Condición de iteración
    while k < maxIter and er_k > tol:
        k += 1 # Aumento de k
        xk = (a + b) / 2.0 # Punto medio del intervalo
        f_xk = f_sym.subs(x, xk).evalf() # Evaluación de f en xk
        er_k = abs(f_xk) # Actualización de error

        # Condición de parada temprana si ya cumple la tolerancia
        if er_k <= tol:
            break

        # Selección del subintervalo que conserva el cambio de signo
        if f_a * f_xk < 0:
            b = xk
            f_b = f_xk
        else:
            a = xk
            f_a = f_xk

    # Condición de convergencia
    if er_k <= tol:
        conv = 1
    else:
        conv = 0

    return xk, er_k, k, conv

##############################################################################
# Método 6: Falsa Posición
##############################################################################

def Falsa_Posicion(f, a, b, tol, maxIter):
    """
    Esta función permite aproximar la solución de ecuaciones no lineales 
    aplicando el método de la Falsa Posición (Regula Falsi).

    Parámetros
    ----------
    f : string 
        Función ingresada en formato texto.
    a : float
        Límite inferior del intervalo [a, b]
    b : float
        Límite superior del intervalo [a, b]
    tol: float
        Tolerancia
    maxIter: integer
        Número máximo de iteraciones

    Retorna
    -------
    xk : float
        Aproximación obtenida
    er_k : float
        Error correspondiente a la última iteración realizada
    k : integer
        Número de iteraciones ejecutadas
    conv : integer
        Variable que indique si el método alcanzó la tolerancia solicitada antes de
        llegar al número máximo de iteraciones
    """
    # Definir x como simbólica
    x = sp.Symbol('x')

    # Conversión de función: Texto a simbólico
    f_sym = sp.sympify(f)

    # Evaluar la función en los extremos del intervalo
    f_a = f_sym.subs(x, a).evalf()
    f_b = f_sym.subs(x, b).evalf()

    # Verificar si alguna de las fronteras ya es raíz
    if f_a == 0:
        return float(a), 0.0, 0, 1
    if f_b == 0:
        return float(b), 0.0, 0, 1

    # Verificar condición de cambio de signo (Teorema de Bolzano)
    if f_a * f_b > 0:
        print("El intervalo [a, b] no cumple la condición de cambio de signo f(a)*f(b) < 0.")
        return None, None, 0, 0

    # Inicialización del conteo de las iteraciones
    k = 0
    
    if (f_b - f_a) == 0:
        print("El denominador se hizo 0. No se puede continuar")
        return None, None, 0, 0

    # Primera aproximación por interpolación lineal (secante)
    xk = b - (f_b * (b - a)) / (f_b - f_a)
    f_xk = f_sym.subs(x, xk).evalf()
    er_k = abs(f_xk)

    # Condición de iteración
    while k < maxIter and er_k > tol:
        k += 1 # Aumento de k

        # Selección del subintervalo que conserva el cambio de signo
        if f_a * f_xk < 0:
            b = xk
            f_b = f_xk
        else:
            a = xk
            f_a = f_xk

        if (f_b - f_a) == 0:
            print("El denominador se hizo 0. No se puede continuar")
            break

        xk = b - (f_b * (b - a)) / (f_b - f_a)
        f_xk = f_sym.subs(x, xk).evalf()
        er_k = abs(f_xk) # Actualización de error

    # Condición de convergencia
    if er_k <= tol:
        conv = 1
    else:
        conv = 0

    return xk, er_k, k, conv