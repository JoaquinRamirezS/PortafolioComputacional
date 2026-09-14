% Cargar funciones de parte1.m
M = parte1();

% Sistema del ejemplo de clase
A = [10  -1   2;
     -1  11  -1;
      2  -1  10];

b = [6; 22; -10];
x0 = [0; 0; 0];
tol = 1e-8;
iterMax = 1000;

% Ejecutar Gauss-Seidel
[xk, erk, k, conv] = M.Gauss_Seidel(A, b, x0, tol, iterMax);

fprintf('=== RESULTADOS GAUSS-SEIDEL ===\n');
fprintf('Aproximación xk:\n'); disp(xk);
fprintf('Error ||Ax-b||_2: %.4e\n', erk);
fprintf('Iteraciones: %d\n', k);
fprintf('Convergencia (conv): %d\n', conv);