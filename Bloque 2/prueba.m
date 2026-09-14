% Cargar métodos de parte1.m
M = parte1();

% Ejemplo de la clase
A = [ 2  -1   0   0;
     -1   2  -1   0;
      0  -1   2  -1;
      0   0  -1   2];

d = [1; 0; 0; 1];

% Resolución con el método de Thomas
x = M.Thomas(A, d);

disp("Solución obtenida:");
disp(x);
