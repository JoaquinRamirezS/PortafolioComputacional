%Cargar métodos de parte1.m
M = parte1();

% Ejemplo de clase
A = [1  1  1  1;
     2  3  1  1;
     1  2  3  1;
     1  1  2  4];

b = [10; 15; 18; 21];

% Ejecutar Factorización LU
x = M.Factorizacion_LU(A, b);

disp("--- Factorización LU ---");
disp("Solución obtenida (x):");
disp(x);
