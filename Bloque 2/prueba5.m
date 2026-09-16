%Cargar métodos de parte1.m
M = parte1();

%Matriz de coeficientes y vector b del ejemplo de clase
A = [1  1  1  1;
     2  3  1  1;
     1  2  3  1;
     1  1  2  4];

b = [10; 15; 18; 21];

%Ejecutar eliminació-gaussiana
x = M.Eliminacion_Gaussiana(A, b);

disp("--- Eliminación Gaussiana ---");
disp("Solución obtenida (x):");
disp(x);
