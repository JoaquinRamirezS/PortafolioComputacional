%Cargar métodos de parte1.m
M = parte1();

% Ejemplo de clase
A = [1  1  0;
     1  0  1;
     0  1  1];

b = [3; 4; 5];

%Ejecutar el método QR
x = M.QR(A, b);

% 3. Mostrar el resultado
disp("--- Solución por Factorización QR ---");
disp(x);

