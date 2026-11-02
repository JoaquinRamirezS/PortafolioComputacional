1; % Instrucción inicial obligatoria para que Octave reconozca el archivo con múltiples funciones

% ===============================================================================
% INSTITUTO TECNOLÓGICO DE COSTA RICA
% CE1111: Análisis Numérico para Ingeniería
% Escuela de Ingeniería en Computadores
%
% Portafolio Computacional - Bloque 4: Parte 1
%
% Autores: Joaquin Ignacio Ramírez Sequeira
%          Joseph Stif Piedra Montero
%
% Implementación de los esquemas numéricos para la Ecuación de Calor 1D:
%   1. Euler Explícito
%   2. Euler Implícito (usando Gradiente Conjugado)
%   3. Crank-Nicolson (usando Gradiente Conjugado)
% ===============================================================================

% ===============================================================================
% 1. ESQUEMA DE EULER EXPLÍCITO
% ===============================================================================
function [x, t, U] = euler_explicito(L, alpha, u0, M, dt, tf)
    % Entrada:
    %   L     : Longitud del dominio espacial [0, L]
    %   alpha : Difusividad térmica (alpha > 0)
    %   u0    : Condición inicial (función anónima u0(x))
    %   M     : Número de subintervalos espaciales (dx = L/M)
    %   dt    : Paso temporal
    %   tf    : Tiempo final
    %
    % Salida:
    %   x     : Malla espacial (M+1 x 1)
    %   t     : Vector de tiempos (N+1 x 1)
    %   U     : Matriz de soluciones (N+1 x M+1) donde U(n+1, i+1) es u_i^n


    % Parámetros de la malla
    dx = L / M;
    x = linspace(0, L, M + 1)';
    
    N = round(tf / dt);
    t = linspace(0, tf, N + 1)';
    
    mu = alpha * dt / (dx^2);
    m = M - 1; % Número de nodos interiores
    
    % Matriz de resultados U (filas = tiempo, columnas = posición)
    U = zeros(N + 1, M + 1);
    
    % Evaluación de la condición inicial
    if isa(u0, 'function_handle')
        u_init = u0(x);
    else
        u_init = feval(u0, x);
    endif
    u_init = u_init(:);
    
    % Condiciones de frontera de Dirichlet nulas: u(0,t) = 0, u(L,t) = 0
    u_init(1) = 0;
    u_init(M + 1) = 0;
    U(1, :) = u_init';
    
    % Construcción de la matriz de avance explícito (I - mu*K)
    % K es la matriz tridiagonal de diferencias finitas
    K = 2 * eye(m) - diag(ones(m - 1, 1), 1) - diag(ones(m - 1, 1), -1);
    M_exp = eye(m) - mu * K;
    
    % Bucle de avance en el tiempo
    u_actual = u_init(2:M); % Nodos interiores en t = 0
    
    for n = 1:N
        u_siguiente = M_exp * u_actual;
        
        % Almacena en U con extremos nulos
        U(n + 1, 2:M) = u_siguiente';
        U(n + 1, 1) = 0;
        U(n + 1, M + 1) = 0;
        
        u_actual = u_siguiente;
    endfor
endfunction

% =============================================================================================
% 2. ESQUEMA DE EULER IMPLÍCITO
% =============================================================================================
function [x, t, U, res_final, conv_all, iter_pasos] = euler_implicito(L, alpha, u0, M, dt, tf)
    % Entrada:
    %   L     : Longitud del dominio espacial [0, L]
    %   alpha : Difusividad térmica (alpha > 0)
    %   u0    : Condición inicial (función anónima u0(x))
    %   M     : Número de subintervalos espaciales (dx = L/M)
    %   dt    : Paso temporal
    %   tf    : Tiempo final
    %
    % Salida:
    %   x, t, U    : Malla espacial, tiempos y matriz de soluciones
    %   res_final  : Vector (N x 1) con los residuos normalizados finales
    %   conv_all   : 1 si todos los sistemas alcanzaron tolerancia, 0 en caso contrario
    %   iter_pasos : Vector (N x 1) con las iteraciones ejecutadas en cada paso temporal
    
    dx = L / M;
    x = linspace(0, L, M + 1)';
    
    N = round(tf / dt);
    t = linspace(0, tf, N + 1)';
    
    mu = alpha * dt / (dx^2);
    m = M - 1;
    
    U = zeros(N + 1, M + 1);
    
    if isa(u0, 'function_handle')
        u_init = u0(x);
    else
        u_init = feval(u0, x);
    endif
    u_init = u_init(:);
    u_init(1) = 0;
    u_init(M + 1) = 0;
    U(1, :) = u_init';
    
    % Construcción de la matriz del sistema implícito: A_E = I + mu * K
    K = 2 * eye(m) - diag(ones(m - 1, 1), 1) - diag(ones(m - 1, 1), -1);
    A_E = eye(m) + mu * K;
    
    tol = 1e-10;
    iterMax = 10000;
    
    res_final = zeros(N, 1);
    iter_pasos = zeros(N, 1);
    conv_all = 1;
    
    u_actual = u_init(2:M);
    
    for n = 1:N
        b = u_actual;  % Lado derecho en Euler implícito
        z0 = u_actual; % Solución del paso anterior como punto de partida
        
        % Resuelve sistema lineal mediante Gradiente Conjugado
        [z_sol, eta, k_iter, conv_step] = resolver_gradiente_conjugado(A_E, b, z0, tol, iterMax);
        
        res_final(n) = eta;
        iter_pasos(n) = k_iter;
        
        if !conv_step
            conv_all = 0;
            printf("Error: El sistema lineal no convergió en el paso temporal n = %d (t = %.4f s)\n", n, t(n + 1));
            break;
        endif
        
        U(n + 1, 2:M) = z_sol';
        U(n + 1, 1) = 0;
        U(n + 1, M + 1) = 0;
        
        u_actual = z_sol;
    endfor
endfunction

% =============================================================================================
% 3. ESQUEMA DE CRANK-NICOLSON
% =============================================================================================
function [x, t, U, res_final, conv_all, iter_pasos] = crank_nicolson(L, alpha, u0, M, dt, tf)
    % Entrada:
    %   L     : Longitud del dominio espacial [0, L]
    %   alpha : Difusividad térmica (alpha > 0)
    %   u0    : Condición inicial (función anónima u0(x))
    %   M     : Número de subintervalos espaciales (dx = L/M)
    %   dt    : Paso temporal
    %   tf    : Tiempo final
    %
    % Salida:
    %   x, t, U    : Malla espacial, tiempos y matriz de soluciones
    %   res_final  : Vector (N x 1) con los residuos normalizados finales
    %   conv_all   : 1 si todos los sistemas alcanzaron tolerancia, 0 en caso contrario
    %   iter_pasos : Vector (N x 1) con las iteraciones ejecutadas en cada paso temporal

    dx = L / M;
    x = linspace(0, L, M + 1)';
    
    N = round(tf / dt);
    t = linspace(0, tf, N + 1)';
    
    mu = alpha * dt / (dx^2);
    m = M - 1;
    
    U = zeros(N + 1, M + 1);
    
    if isa(u0, 'function_handle')
        u_init = u0(x);
    else
        u_init = feval(u0, x);
    endif
    u_init = u_init(:);
    u_init(1) = 0;
    u_init(M + 1) = 0;
    U(1, :) = u_init';
    
    % Construcción de matrices Crank-Nicolson: A_C * u^{n+1} = B_C * u^n
    K = 2 * eye(m) - diag(ones(m - 1, 1), 1) - diag(ones(m - 1, 1), -1);
    A_C = eye(m) + (mu / 2) * K;
    B_C = eye(m) - (mu / 2) * K;
    
    tol = 1e-10;
    iterMax = 10000;
    
    res_final = zeros(N, 1);
    iter_pasos = zeros(N, 1);
    conv_all = 1;
    
    u_actual = u_init(2:M);
    
    for n = 1:N
        b = B_C * u_actual; % Lado derecho en Crank-Nicolson
        z0 = u_actual;      % Aproximación inicial del paso anterior
        
        % Resuelve sistema lineal mediante Gradiente Conjugado
        [z_sol, eta, k_iter, conv_step] = resolver_gradiente_conjugado(A_C, b, z0, tol, iterMax);
        
        res_final(n) = eta;
        iter_pasos(n) = k_iter;
        
        if !conv_step
            conv_all = 0;
            printf("Error: El sistema lineal no convergió en el paso temporal n = %d (t = %.4f s)\n", n, t(n + 1));
            break;
        endif
        
        U(n + 1, 2:M) = z_sol';
        U(n + 1, 1) = 0;
        U(n + 1, M + 1) = 0;
        
        u_actual = z_sol;
    endfor
endfunction

% =============================================================================================
% FUNCIÓN AUXILIAR: GRADIENTE CONJUGADO OPTIMIZADO
% =============================================================================================
function [z, eta_final, k, conv] = resolver_gradiente_conjugado(A, b, z0, tol, iterMax)
    % Resuelve A*z = b asegurando tolerancia y recalculando el residuo verdadero
    b = b(:);
    z = z0(:);
    
    norm_b = max(1, norm(b, 2));
    r = b - A * z;
    p = r;
    k = 0;
    
    eta = norm(r, 2) / norm_b;
    
    if eta < tol
        conv = 1;
        eta_final = eta;
        return;
    endif
    
    while (eta >= tol) && (k < iterMax)
        Ap = A * p;
        r_sq = r' * r;
        alpha_step = r_sq / (p' * Ap);
        
        z = z + alpha_step * p;
        r_next = r - alpha_step * Ap;
        
        % Evaluación rápida del residuo relativo con r_next (O(m))
        eta = norm(r_next, 2) / norm_b;
        
        if eta < tol
            k = k + 1;
            break;
        endif
        
        beta_step = (r_next' * r_next) / r_sq;
        p = r_next + beta_step * p;
        r = r_next;
        
        k = k + 1;
    endwhile
    
    % Comprobación final del residuo verdadero b - A*z para la tolerancia
    r_true = b - A * z;
    eta_final = norm(r_true, 2) / norm_b;
    conv = (eta_final < tol);
endfunction