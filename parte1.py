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
