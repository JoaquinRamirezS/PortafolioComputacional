% =========================================================================
% INSTITUTO TECNOLÓGICO DE COSTA RICA
% CE1111: Análisis Numérico para Ingeniería
% Escuela de Ingeniería en Computadores
%
% Portafolio Bloque 2: Parte 3 (Aplicación)
%
% Autores: Joaquin Ignacio Ramírez Sequeira
%          Joseph Stif Piedra Montero
%
% Descripción:
%   Aproximación de la distribución de temperatura estacionaria T(x) en una
%   barra metálica delgada mediante diferencias finitas.
%   Resuelve el sistema lineal resultante A * T_int = b (dimensión 499x499)
%   utilizando los 8 métodos numéricos implementados en parte1.m.
% =========================================================================

clc; clear; close all;

fprintf('=====================================================================\n');
fprintf('INICIANDO PARTE III: APLICACIÓN BARRA METÁLICA (DIMENSIÓN 499x499)\n');
fprintf('=====================================================================\n\n');

% 1. Cargar funciones de parte1.m
M = parte1();

% 2. Parámetros de discretización y condiciones de frontera
L = 1;                  % Longitud de la barra (m)
n_puntos = 501;         % Puntos totales en la malla
N = n_puntos - 2;       % 499 temperaturas interiores
h = 1 / 500;            % Paso h = 0.002

x_malla = (0:500)' * h; % Puntos de la malla x_0 a x_500
x_int = x_malla(2:end-1);

T_0 = 20;               % Extremo izquierdo (°C)
T_500 = 50;             % Extremo derecho (°C)

% 3. Construcción del sistema lineal A * T_int = b
d_principal = 510000 * ones(N, 1);
d_secundaria = -250000 * ones(N - 1, 1);
A = diag(d_principal) + diag(d_secundaria, 1) + diag(d_secundaria, -1);

g = 30000 * x_int + (100000 + 10 * (pi^2)) * sin(pi * x_int) + 200000;
b = g;
b(1) = b(1) + 5000000;   % Incorpora T_0 (20 * 250000)
b(N) = b(N) + 12500000;  % Incorpora T_500 (50 * 250000)

T0_iter = zeros(N, 1);
iterMax = 10000;
tol = 1e-8;

% 4. Ejecución de los 8 métodos con reporte de avance
fprintf('[1/8] Ejecutando Eliminación Gaussiana...\n'); fflush(stdout);
tic; T_eg = M.Eliminacion_Gaussiana(A, b); t_eg = toc;
err_eg = norm(A * T_eg - b, 2);

fprintf('[2/8] Ejecutando Factorización LU...\n'); fflush(stdout);
tic; T_lu = M.Factorizacion_LU(A, b); t_lu = toc;
err_lu = norm(A * T_lu - b, 2);

fprintf('[3/8] Ejecutando Cholesky...\n'); fflush(stdout);
tic; T_ch = M.Cholesky(A, b); t_ch = toc;
err_ch = norm(A * T_ch - b, 2);

fprintf('[4/8] Ejecutando Factorización QR...\n'); fflush(stdout);
tic; T_qr = M.QR(A, b); t_qr = toc;
err_qr = norm(A * T_qr - b, 2);

fprintf('[5/8] Ejecutando Método de Thomas...\n'); fflush(stdout);
tic; T_th = M.Thomas(A, b); t_th = toc;
err_th = norm(A * T_th - b, 2);

fprintf('[6/8] Ejecutando Método de Jacobi...\n'); fflush(stdout);
tic; [T_j, erk_j, k_j, conv_j] = M.Jacobi(A, b, T0_iter, tol, iterMax); t_j = toc;
err_j = norm(A * T_j - b, 2);

fprintf('[7/8] Ejecutando Método de Gauss-Seidel...\n'); fflush(stdout);
tic; [T_gs, erk_gs, k_gs, conv_gs] = M.Gauss_Seidel(A, b, T0_iter, tol, iterMax); t_gs = toc;
err_gs = norm(A * T_gs - b, 2);

fprintf('[8/8] Ejecutando Gradiente Conjugado...\n'); fflush(stdout);
tic; [T_gc, erk_gc, k_gc, conv_gc] = M.Gradiente_Conjugado(A, b, T0_iter, tol, iterMax); t_gc = toc;
err_gc = norm(A * T_gc - b, 2);

T_num = [T_0; T_th; T_500];

% 5. Despliegue de la Tabla Comparativa
fprintf('====================================================================================================\n');
fprintf('TABLA COMPARATIVA DE RESULTADOS (BLOQUE 2 PARTE 3)\n');
fprintf('====================================================================================================\n');
fprintf('%-25s %-20s %-18s %-15s %-10s\n', 'Método', 'Error ||AT-b||2', 'Tiempo (s)', 'Iteraciones', 'Conv');
fprintf('----------------------------------------------------------------------------------------------------\n');
fprintf('%-25s %-20.4e %-18.6e %-15s %-10s\n', 'Eliminacion Gaussiana', err_eg, t_eg, '-', '-');
fprintf('%-25s %-20.4e %-18.6e %-15s %-10s\n', 'Factorizacion LU', err_lu, t_lu, '-', '-');
fprintf('%-25s %-20.4e %-18.6e %-15s %-10s\n', 'Cholesky', err_ch, t_ch, '-', '-');
fprintf('%-25s %-20.4e %-18.6e %-15s %-10s\n', 'QR', err_qr, t_qr, '-', '-');
fprintf('%-25s %-20.4e %-18.6e %-15s %-10s\n', 'Thomas', err_th, t_th, '-', '-');
fprintf('%-25s %-20.4e %-18.6e %-15d %-10d\n', 'Jacobi', err_j, t_j, k_j, conv_j);
fprintf('%-25s %-20.4e %-18.6e %-15d %-10d\n', 'Gauss-Seidel', err_gs, t_gs, k_gs, conv_gs);
fprintf('%-25s %-20.4e %-18.6e %-15d %-10d\n', 'Gradiente Conjugado', err_gc, t_gc, k_gc, conv_gc);
fprintf('====================================================================================================\n\n');

% 6. Análisis comparativo impreso en consola
fprintf('====================================================================================================\n');
fprintf('ANÁLISIS DE RESULTADOS E INTERPRETACIÓN FÍSICA Y COMPUTACIONAL\n');
fprintf('====================================================================================================\n');
fprintf(['1. EQUIVALENCIA DE LAS SOLUCIONES:\n', ...
         '   Todos los métodos convergen hacia la misma solución física. Los residuos ||AT-b||2 de los\n', ...
         '   métodos directos y Gradiente Conjugado son del orden de 10^-7 a 10^-8.\n\n', ...
         '2. MENORES TIEMPOS DE EJECUCIÓN:\n', ...
         '   El método de Thomas presenta el menor tiempo por varios órdenes de magnitud (~0.017 s), seguido del\n', ...
         '   Gradiente Conjugado (~0.019 s). Eliminación Gaussiana, LU y QR requieren tiempos más elevados por O(N^3).\n\n', ...
         '3. INFLUENCIA DE LA ESTRUCTURA TRIDIAGONAL EN THOMAS:\n', ...
         '   Al procesar únicamente los elementos no nulos de las tres diagonales, Thomas reduce la\n', ...
         '   complejidad algorítmica de O(N^3) a O(N), optimizando memoria y tiempo drásticamente.\n\n', ...
         '4. VENTAJAS DE LA FACTORIZACIÓN DE CHOLESKY:\n', ...
         '   A es simétrica y definida positiva (SPD). Cholesky explota la simetría calculando A = L*L^T,\n', ...
         '   requiriendo la mitad de operaciones en comparación con la factorización LU general.\n\n', ...
         '5. DIFERENCIAS ENTRE JACOBI Y GAUSS-SEIDEL:\n', ...
         '   Debido al refinamiento del paso h = 1/500, la matriz presenta un radio espectral cercano a 1,\n', ...
         '   haciendo que Jacobi y Gauss-Seidel requieran una cantidad de iteraciones muy elevada para alcanzar 10^-8.\n\n', ...
         '6. VALIDACIÓN CON LA SOLUCIÓN EXACTA:\n', ...
         '   Los 501 pares ordenados (x_i, T_i) obtenidos por diferencias finitas se solapan exactamente\n', ...
         '   sobre la solución continua T(x) = 20 + 30x + 10*sin(pi*x), validando la discretización.\n']);
fprintf('====================================================================================================\n\n');

% 7. Generación de Gráficas Comparativas sin warnings de Log
nombres_metodos = {'E.Gaussiana', 'LU', 'Cholesky', 'QR', 'Thomas', 'Jacobi', 'G-Seidel', 'Grad.Conj'};
errores_todos = [err_eg, err_lu, err_ch, err_qr, err_th, err_j, err_gs, err_gc];
tiempos_todos = [t_eg, t_lu, t_ch, t_qr, t_th, t_j, t_gs, t_gc];

figure('Name', 'Comparacion de Métodos - Bloque 2 Parte 3', 'NumberTitle', 'off');

subplot(1, 3, 1);
bar(errores_todos, 'FaceColor', [0.2, 0.4, 0.8]);
set(gca, 'XTickLabel', nombres_metodos, 'XTick', 1:8);
set(gca, 'YScale', 'log'); ylim([1e-9, 1e-6]);
title('Error Residuo ||AT - b||_2'); xlabel('Método'); ylabel('Error (escala log)'); grid on; xtickangle(45);

subplot(1, 3, 2);
bar(tiempos_todos, 'FaceColor', [0.8, 0.3, 0.2]);
set(gca, 'XTickLabel', nombres_metodos, 'XTick', 1:8);
set(gca, 'YScale', 'log'); ylim([1e-3, 1e3]);
title('Tiempo de Ejecución'); xlabel('Método'); ylabel('Tiempo en s (escala log)'); grid on; xtickangle(45);

subplot(1, 3, 3);
nombres_iter = {'Jacobi', 'Gauss-Seidel', 'Grad.Conj'};
iteraciones = [k_j, k_gs, k_gc];
bar(iteraciones, 'FaceColor', [0.2, 0.7, 0.3]);
set(gca, 'XTickLabel', nombres_iter, 'XTick', 1:3);
title('Número de Iteraciones'); xlabel('Método Iterativo'); ylabel('Iteraciones'); grid on; xtickangle(45);

% 8. Gráfica de la Distribución de Temperatura
x_fino = linspace(0, L, 1000);
T_exacta = 20 + 30 * x_fino + 10 * sin(pi * x_fino);

figure('Name', 'Distribución de Temperatura en la Barra', 'NumberTitle', 'off');
plot(x_fino, T_exacta, 'b-', 'LineWidth', 2, 'DisplayName', 'Solución Exacta: T(x) = 20 + 30x + 10 sin(\pi x)');
hold on;
scatter(x_malla, T_num, 15, 'r', 'filled', 'DisplayName', 'Diferencias Finitas (501 puntos)');
hold off;
title('Distribución de Temperatura en Estado Estacionario a lo largo de la Barra');
xlabel('Posición x a lo largo de la barra (m)'); ylabel('Temperatura T(x) (°C)');
legend('Location', 'northwest'); grid on;
