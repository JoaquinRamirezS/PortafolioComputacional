% Cargar métodos de parte1.m
M = parte1();

% Ejemplo de clase (HOJA 3)
A = [4  2  2;
     2  5  1;
     2  1  6];

b = [8; 8; 9];

% Ejecutar el método Cholesky
x = M.Cholesky(A, b);

% Mostrar el resultado
disp("--- Solución por Factorización Cholesky ---");
disp(x);
