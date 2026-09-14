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