M = parte1();

A = [4  1  0  0;
     1  4  1  0;
     0  1  4  1;
     0  0  1  4];

b = [5; 6; 6; 5];
x0 = [0; 0; 0; 0];
tol = 1e-8;
iterMax = 1000;

[xk, erk, k, conv] = M.Gradiente_Conjugado(A, b, x0, tol, iterMax);

fprintf('=== RESULTADOS GRADIENTE CONJUGADO ===\n');
fprintf('Aproximación xk:\n'); disp(xk);
fprintf('Error ||b - Ax||_2: %.4e\n', erk);
fprintf('Iteraciones: %d\n', k);
fprintf('Convergencia (conv): %d\n', conv);