% ===============================================================================
% INSTITUTO TECNOLÓGICO DE COSTA RICA
% CE1111: Análisis Numérico para Ingeniería
% Escuela de Ingeniería en Computadores
%
% Portafolio Bloque 2: Parte 1
%
% Autores: Joaquin Ignacio Ramírez Sequeira
% Joseph Stif Piedra Montero
%
% Este archivo contiene la implementación computacional de los métodos
% para la solución de sistemas de ecuaciones lineales Ax = b.
% =========================================================================

function M = parte1()
    % Retorna la estructura con las referencias a los métodos implementados
    M.Thomas = @Thomas;
    M.Jacobi = @Jacobi;
    M.Gauss_Seidel = @Gauss_Seidel;
    M.Gradiente_Conjugado = @Gradiente_Conjugado;
    % Se irán agregando los demás métodos: Jacobi, GaussSeidel, GradienteConjugado...
endfunction

% =========================================================================
% Método de Thomas para Matrices Tridiagonales
% =========================================================================
function x = Thomas(A, b)
    % Entrada:
    %   A : Matriz tridiagonal de coeficientes de tamaño (n x n)
    %   b : Vector de términos independientes de tamaño (n x 1)
    %
    % Salida:
    %   x : Vector solución de tamaño (n x 1)

    n = length(b);
    b = b(:); % Garantizar vector columna

    % Extracción de las diagonales de la matriz A
    % b_diag : diagonal principal (b_1, ..., b_n)
    % a_sub  : subdiagonal (a_2, ..., a_n), con a_1 = 0
    % c_sup  : superdiagonal (c_1, ..., c_{n-1}), con c_n = 0
    b_diag = diag(A);
    a_sub  = [0; diag(A, -1)];
    c_sup  = [diag(A, 1); 0];
    d      = b;

    % Inicialización de los vectores auxiliares p y q
    p = zeros(n, 1);
    q = zeros(n, 1);

    % ---------------------------------------------------------------------
    % FASE DE ELIMINACIÓN FORWARD
    % ---------------------------------------------------------------------
    % 1. Inicialización para i = 1
    p(1) = c_sup(1) / b_diag(1);
    q(1) = d(1) / b_diag(1);

    % 2 y 3. Iteraciones para i = 2 hasta n-1
    for i = 2:n-1
        denom = b_diag(i) - p(i-1) * a_sub(i);
        p(i) = c_sup(i) / denom;
        q(i) = (d(i) - q(i-1) * a_sub(i)) / denom;
    endfor

    % 3. Último cálculo de q_n para i = n
    denom = b_diag(n) - p(n-1) * a_sub(n);
    q(n) = (d(n) - q(n-1) * a_sub(n)) / denom;

    % ---------------------------------------------------------------------
    % FASE DE SUSTITUCIÓN BACKWARD
    % ---------------------------------------------------------------------
    x = zeros(n, 1);

    % 4. Último valor del vector solución
    x(n) = q(n);

    % 5. Sustitución hacia atrás para i = n-1 descendiendo hasta 1
    for i = n-1:-1:1
        x(i) = q(i) - p(i) * x(i+1);
    endfor
endfunction

% =========================================================================
% Método Iterativo de Jacobi
% =========================================================================
function [xk, erk, k, conv] = Jacobi(A, b, x0, tol, iterMax)
    % Esta función aproxima la solución del sistema lineal Ax = b mediante
    % el método iterativo de Jacobi.
    %
    % Parámetros
    % ----------
    % A       : matrix (n x n) - Matriz de coeficientes del sistema.
    % b       : vector (n x 1) - Vector de términos independientes.
    % x0      : vector (n x 1) - Vector de aproximación inicial x^(0).
    % tol     : float          - Tolerancia para la norma 2 del residuo.
    % iterMax : integer        - Número máximo de iteraciones permitidas.
    %
    % Retorna
    % -------
    % xk   : vector (n x 1) - Aproximación obtenida.
    % erk  : float          - Error norma 2 del residuo ||A*xk - b||_2.
    % k    : integer        - Número de iteraciones ejecutadas.
    % conv : integer        - Indicador de convergencia (1 si convergió, 0 si no).

    % Paso 1: Número de filas/columnas de A
    b = b(:); % Garantizar vector columna
    n = length(b);

    % Paso 2: Recíprocos de los elementos diagonales de A (vector d_inv)
    d_inv = 1 ./ diag(A);

    % Paso 3: Descomposición R = L + U (matriz A sin la diagonal principal)
    D = diag(diag(A));
    R = A - D;

    % Paso 4: Inicialización de variables
    xk = x0(:); % Asegurar vector columna
    erk = norm(A * xk - b, 2);
    k = 0;

    % Paso 5: Ciclo iterativo
    while (erk > tol) && (k < iterMax)
        % Cálculo de x^(k+1) = D^(-1) * (b - R * x^(k)) mediante producto de Hadamard
        x_nuevo = d_inv .* (b - R * xk);
        
        % Actualización de la aproximación y del residuo
        xk = x_nuevo;
        erk = norm(A * xk - b, 2);
        k = k + 1;
    endwhile

    % Paso 6: Verificación del criterio de convergencia
    if erk < tol
        conv = 1;
    else
        conv = 0;
    endif
endfunction

% =========================================================================
% Método Iterativo de Gauss-Seidel
% =========================================================================
function [xk, erk, k, conv] = Gauss_Seidel(A, b, x0, tol, iterMax)
    % Esta función aproxima la solución del sistema lineal Ax = b mediante
    % el método iterativo de Gauss-Seidel resolviendo M*x^(k+1) = c mediante
    % sustitución hacia adelante (donde M = L + D).
    %
    % Parámetros
    % ----------
    % A       : matrix (n x n) - Matriz de coeficientes del sistema.
    % b       : vector (n x 1) - Vector de términos independientes.
    % x0      : vector (n x 1) - Vector de aproximación inicial x^(0).
    % tol     : float          - Tolerancia para la norma 2 del residuo.
    % iterMax : integer        - Número máximo de iteraciones permitidas.
    %
    % Retorna
    % -------
    % xk   : vector (n x 1) - Aproximación obtenida.
    % erk  : float          - Error norma 2 del residuo ||A*xk - b||_2.
    % k    : integer        - Número de iteraciones ejecutadas.
    % conv : integer        - Indicador de convergencia (1 si convergió, 0 si no).

    % Paso 1: Obtener L, D y U de la matriz A
    b = b(:);    
    D = diag(diag(A));
    L = tril(A, -1);
    U = triu(A, 1);

    % Paso 2: Matriz triangular inferior M = L + D
    M = L + D;

    % Paso 3: Inicialización de variables
    xk = x0(:);
    erk = norm(A * xk - b, 2);
    k = 0;

    % Paso 4: Ciclo iterativo
    while (erk > tol) && (k < iterMax)
        % Término independiente c = b - U * x^(k)
        c = b - U * xk;

        % Resolver M * x_nuevo = c mediante sustitución hacia adelante
        x_nuevo = sust_adelante(M, c);

        % Actualización de la aproximación y del residuo
        xk = x_nuevo;
        erk = norm(A * xk - b, 2);
        k = k + 1;
    endwhile

    % Paso 5: Verificación del criterio de convergencia
    if erk < tol
        conv = 1;
    else
        conv = 0;
    endif
endfunction

% =========================================================================
% Función Auxiliar: Sustitución hacia adelante para M*y = c
% =========================================================================
function y = sust_adelante(M, c)
    n = length(c);
    y = zeros(n, 1);
    for i = 1:n
        s = 0;
        for j = 1:(i - 1)
            s = s + M(i, j) * y(j);
        endfor
        y(i) = (c(i) - s) / M(i, i);
    endfor
endfunction

% =========================================================================
% Método Iterativo del Gradiente Conjugado
% =========================================================================
function [xk, erk, k, conv] = Gradiente_Conjugado(A, b, x0, tol, iterMax)
    % Entrada:
    %   A       : Matriz simétrica y definida positiva (SPD) (n x n)
    %   b       : Vector de términos independientes (n x 1)
    %   x0      : Vector de aproximación inicial (n x 1)
    %   tol     : Tolerancia para la norma 2 del residuo
    %   iterMax : Número máximo de iteraciones permitidas
    %
    % Salida:
    %   xk   : Vector solución aproximado
    %   erk  : Error norma 2 del residuo ||b - A*xk||_2
    %   k    : Número de iteraciones ejecutadas
    %   conv : 1 si convergió (erk < tol), 0 si no

    % Paso 1: Garantizar vectores columna
    b = b(:);
    xk = x0(:);

    % Paso 2: Inicialización 
    rk = b - A * xk;
    pk = rk;
    k = 0;
    erk = norm(rk, 2);

    % Paso 3: Ciclo iterativo 
    while (norm(rk, 2) >= tol) && (k < iterMax)
        Apk = A * pk;
        rkrk = rk' * rk;
        alpha = rkrk / (pk' * Apk);

        % Actualización de solución y residuo 
        xk = xk + alpha * pk;
        rk_next = rk - alpha * Apk;
        
        % Verificación de convergencia intermedia 
        if norm(rk_next, 2) < tol
            k = k + 1;
            break;
        endif

        % Factor beta y nueva dirección de búsqueda 
        beta = (rk_next' * rk_next) / rkrk;
        pk = rk_next + beta * pk;
        rk = rk_next;
        
        k = k + 1; 
    endwhile

    % Paso 4: Cálculo final de error y conv 
    erk = norm(b - A * xk, 2);
    if erk < tol
        conv = 1;
    else
        conv = 0;
    endif
endfunction