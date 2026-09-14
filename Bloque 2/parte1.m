% =========================================================================
% INSTITUTO TECNOLÓGICO DE COSTA RICA
% Escuela de Matemática / Escuela de Ingeniería en Computadores
# CE1111 - Análisis Numérico para Ingeniería
%
% Archivo: parte1.m (Sección: Método de Thomas)
% Autores: Joaquin Ignacio Ramírez Sequeira
%          Joseph Stif Piedra Montero
% =========================================================================

function M = parte1()
    % Retorna la estructura con las referencias a los métodos implementados
    M.Thomas = @Thomas;
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