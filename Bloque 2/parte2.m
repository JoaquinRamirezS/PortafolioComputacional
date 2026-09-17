% =========================================================================
% INSTITUTO TECNOLÓGICO DE COSTA RICA
% CE1111: Análisis Numérico para Ingeniería
% Escuela de Ingeniería en Computadores
%
% Portafolio Bloque 2: Parte 2
%
% Autores: Joaquin Ignacio Ramírez Sequeira
% Joseph Stif Piedra Montero
%
% Comparación computacional de los métodos por medio del cálculo del error
% y la toma de tiempos de ejecución.
% =========================================================================

clc; clear; close all;
%---------------------------------------------------------------------------
%Generación de la matriz A y el vector b
%---------------------------------------------------------------------------
%Tamaño
n = 300;

%Matriz A
A=4*eye(n) - diag(ones(n-1,1),1)- diag(ones(n-1,1),-1);
%Vector b
b = ones(n,1);

%Parámetros requeridos en los metodos iterativos
x0=zeros(n,1);
iterMax=10000;
tol= 1e-8;

%-------------------------------------------------------------------------
%Importar métodos de parte1
%-------------------------------------------------------------------------
M = parte1();

%---------------------------------------------------------------------------
%Eliminación Gaussiana
%---------------------------------------------------------------------------
%Tiempo de ejecución
tic; x_eg = M.Eliminacion_Gaussiana(A,b); time_eg = toc;
%Cálculo del error
err_eg = norm(A*x_eg-b,2);

%---------------------------------------------------------------------------
%Factorización LU
%---------------------------------------------------------------------------
%Tiempo de ejecución
tic; x_lu = M.Factorizacion_LU(A,b); time_lu = toc;
%Cálculo del error
err_lu = norm(A*x_lu-b,2);

%---------------------------------------------------------------------------
%Cholesky
%---------------------------------------------------------------------------
%Tiempo de ejecución
tic; x_ch = M.Cholesky(A,b); time_ch = toc;
%Cálculo del error
err_ch = norm(A*x_ch-b,2);

%---------------------------------------------------------------------------
%QR
%---------------------------------------------------------------------------
%Tiempo de ejecución
tic; x_qr = M.QR(A,b); time_qr = toc;
%Cálculo del error
err_qr = norm(A*x_qr-b,2);

%---------------------------------------------------------------------------
%Thomas
%---------------------------------------------------------------------------
%Tiempo de ejecución
tic; x_th = M.Thomas(A,b); time_th = toc;
%Cálculo del error
err_th = norm(A*x_th-b,2);

%---------------------------------------------------------------------------
%Jacobi
%---------------------------------------------------------------------------
%Tiempo de ejecución
tic;
[x_j,erk_j,k_j,conv_j] = M.Jacobi(A,b,x0,tol,iterMax);
time_j = toc;
%Cálculo del error
err_j = norm(A*x_j-b,2);

%---------------------------------------------------------------------------
%Gauss_Seidel
%---------------------------------------------------------------------------
%Tiempo de ejecución
tic;
[x_gs,erk_gs,k_gs,conv_gs] = M.Gauss_Seidel(A,b,x0,tol,iterMax);
time_gs = toc;
%Cálculo del error
err_gs = norm(A*x_gs-b,2);

%---------------------------------------------------------------------------
%Gradiente Conjugado
%---------------------------------------------------------------------------
%Tiempo de ejecución
tic;
[x_gc,erk_gc,k_gc,conv_gc] = M.Gradiente_Conjugado(A,b,x0,tol,iterMax);
time_gc = toc;
%Cálculo del error
err_gc = norm(A*x_gc-b,2);

%---------------------------------------------------------------------------
%Tabla comparativa
%---------------------------------------------------------------------------

%---------------------------------------------------------------------------
%Tabla comparativa
%---------------------------------------------------------------------------
fprintf('\n');
fprintf('==========================================================================================\n');
fprintf('                              TABLA COMPARATIVA DE MÉTODOS\n');
fprintf('==========================================================================================\n');
fprintf('%-22s | %-14s | %-12s | %-10s | %-6s\n', ...
        'Método', 'Error ||Ax-b||', 'Tiempo (s)', 'Iter (k)', 'Conv');
fprintf('------------------------------------------------------------------------------------------\n');

%Métodos directos
fprintf('%-22s | %-14.4e | %-12.6f | %-10s | %-6s\n', ...
        'Eliminación Gaussiana', err_eg, time_eg, '-', '-');
fprintf('%-22s | %-14.4e | %-12.6f | %-10s | %-6s\n', ...
        'Factorización LU', err_lu, time_lu, '-', '-');
fprintf('%-22s | %-14.4e | %-12.6f | %-10s | %-6s\n', ...
        'Cholesky', err_ch, time_ch, '-', '-');
fprintf('%-22s | %-14.4e | %-12.6f | %-10s | %-6s\n', ...
        'QR', err_qr, time_qr, '-', '-');
fprintf('%-22s | %-14.4e | %-12.6f | %-10s | %-6s\n', ...
        'Thomas', err_th, time_th, '-', '-');

% Métodos iterativos
fprintf('%-22s | %-14.4e | %-12.6f | %-10d | %-6d\n', ...
        'Jacobi', err_j, time_j, k_j, conv_j);
fprintf('%-22s | %-14.4e | %-12.6f | %-10d | %-6d\n', ...
        'Gauss-Seidel', err_gs, time_gs, k_gs, conv_gs);
fprintf('%-22s | %-14.4e | %-12.6f | %-10d | %-6d\n', ...
        'Gradiente Conjugado', err_gc, time_gc, k_gc, conv_gc);

fprintf('==========================================================================================\n\n');











