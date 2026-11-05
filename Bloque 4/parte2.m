% ===============================================================================
% INSTITUTO TECNOLÓGICO DE COSTA RICA
% CE1111: Análisis Numérico para Ingeniería
% Escuela de Ingeniería en Computadores
%
% Portafolio Computacional - Bloque 4: Parte 2
%
% Autores: Joaquin Ignacio Ramírez Sequeira
%          Joseph Stif Piedra Montero
%
% Este script ejecuta los experimentos de la Parte II: validación de los
% esquemas, refinamiento de malla, estabilidad, tiempo de enfriamiento y
% mapa de calor. Utiliza los métodos numéricos cargados desde parte1.m.
% ===============================================================================
function parte2()
clc;close all;
%Funciones parte1
source('parte1.m');

%Crear carpeta para guardar las gráficas
if ~exist('graficas', 'dir')
    mkdir('graficas');
end

%Parametros del problema
L = 1.0;      %Longitud
alpha = 0.01; %Difusividad térmica
M = 50;       %Número de subintervalo espaciales
dt = 0.01;    %Paso temporal
tf = 10.0;    %Tiempo final

%Condicion inicial para u = T - 20
u0 = @(x) 80 * sin(pi * x / L);

%Solucion exacta de T solo para calc errores
T_exacta = @(x,t) 20 + 80 * exp(-alpha*(pi/L)^2*t) .* sin(pi*x/L);

fprintf('\n============================================================\n');
fprintf(' BLOQUE 4 - PARTE II: ENFRIAMIENTO DE UNA BARRA\n');
fprintf('============================================================\n');
fprintf('L = %.2f m, alpha = %.4f m^2/s\n', L, alpha);
fprintf('M = %d, dt = %.4f s, tf = %.2f s\n', M, dt, tf);

% Verificar si Euler explicito cumple con la condicion de estabilidad
dx = L/M;
mu = alpha*dt/dx^2;
fprintf('\n--- Verificacion de estabilidad del caso principal ---\n');
fprintf('dx = %.6f m, mu = alpha*dt/dx^2 = %.6f\n', dx, mu);
if mu <= 0.5
    fprintf('Euler explicito cumple mu <= 0.5: ESTABLE.\n');
else
    fprintf('Euler explicito no cumple mu <= 0.5: INESTABLE.\n');
end

%Experimento principal (contempla de puntos 3 a 7)
resultados = experimento_principal(L, alpha, u0, T_exacta, M, dt, tf);

%Refinamiento de malla (punto 8)
experimento_refinamiento(L, alpha, u0, T_exacta);

% Estabilidad de euler explicito (punto 9)
experimento_estabilidad(L, alpha, M);

%Tiempo de enfriamiento(punto 10 y 11)
experimento_enfriamiento(L, alpha, u0, M, dt);

%Mapa de calor (punto 12)
experimento_mapa_calor(L, alpha, u0, M, dt);

%Resumen final (punto 13)
fprintf('\n============================================================\n');
fprintf(' RESUMEN FINAL\n');
fprintf('============================================================\n');
fprintf('Error final: Euler explicito = %.4e\n', resultados.Ef(1));
fprintf('Error final: Euler implicito = %.4e\n', resultados.Ef(2));
fprintf('Error final: Crank-Nicolson = %.4e\n', resultados.Ef(3));
fprintf('Tiempo: Euler explicito = %.4f s\n', resultados.tiempo(1));
fprintf('Tiempo: Euler implicito = %.4f s\n', resultados.tiempo(2));
fprintf('Tiempo: Crank-Nicolson = %.4f s\n', resultados.tiempo(3));


fprintf('\n============================================================\n');
fprintf(' ANALISIS E INTERPRETACION DE RESULTADOS\n');
fprintf('============================================================\n');

fprintf('\n1. Precision y costo computacional:\n');
fprintf('   Los tres metodos dieron buenos resultados. Euler explicito fue el\n');
fprintf('   mas rapido, mientras que Crank-Nicolson obtuvo mayor precision que\n');
fprintf('   Euler implicito. Sin embargo, Euler explicito debe cumplir una\n');
fprintf('   condicion de estabilidad para funcionar correctamente.\n');

fprintf('\n2. Estabilidad y refinamiento:\n');
fprintf('   Al reducir el tamaño de la malla espacial, el error disminuyo.\n');
fprintf('   En cambio, al reducir el paso temporal, el error no siempre bajo,\n');
fprintf('   ya que el error espacial tambien influye en los resultados.\n');

fprintf('\n3. Temperatura y tiempo de enfriamiento:\n');
fprintf('   La temperatura maxima se encuentra en el centro de la barra y\n');
fprintf('   disminuye conforme pasa el tiempo. El tiempo numerico para llegar\n');
fprintf('   a 30 grados fue cercano al valor analitico de 21.069 segundos.\n');

fprintf('\n4. Condiciones de frontera y aislamiento:\n');
fprintf('   Los extremos se mantienen a 20 grados, permitiendo que el calor\n');
fprintf('   se transfiera hacia ellos. Como los lados estan aislados, el calor\n');
fprintf('   solo se estudia en una direccion y el enfriamiento es simetrico.\n');
fprintf('\n============================================================\n');
end
% Definicion de funciones
function resultados = experimento_principal(L, alpha, u0, T_exacta, M, dt, tf)
    fprintf('\n=== EXPERIMENTO PRINCIPAL ===\n');

    %Euler explicito
    tic;
    [x, t, U_exp] = euler_explicito(L, alpha, u0, M, dt, tf);
    tiempo_exp = toc;
    T_exp = U_exp + 20;

    %Euler implicito
    tic;
    [~, ~, U_imp, res_imp, conv_imp, iter_imp] = ...
        euler_implicito(L, alpha, u0, M, dt, tf);
    tiempo_imp = toc;
    T_imp = U_imp + 20;

    %Crank-Nicolson
    tic;
    [~, ~, U_cn, res_cn, conv_cn, iter_cn] = ...
        crank_nicolson(L, alpha, u0, M, dt, tf);
    tiempo_cn = toc;
    T_cn = U_cn + 20;

    %Errores finales y maximos sobre toda la malla
    [Ef_exp, Emalla_exp] = calcular_errores(T_exp, x, t, T_exacta);
    [Ef_imp, Emalla_imp] = calcular_errores(T_imp, x, t, T_exacta);
    [Ef_cn, Emalla_cn] = calcular_errores(T_cn, x, t, T_exacta);

    %Tabla comparativa
    fprintf('\nTabla comparativa del experimento principal\n');
    fprintf('%-18s | %11s | %11s | %10s | %12s | %10s | %10s | %s\n', ...
        'Metodo','Ef','Emalla','Tiempo(s)','Max residuo','Iter. acum.','Max iter.','Convergencia');
    fprintf('%s\n', repmat('-',1,119));

    imprimir_fila('Euler explicito', Ef_exp, Emalla_exp, tiempo_exp, ...
        NaN, NaN, NaN, 'No aplica');
    imprimir_fila('Euler implicito', Ef_imp, Emalla_imp, tiempo_imp, ...
        max(res_imp), sum(iter_imp), max(iter_imp), estado_convergencia(conv_imp));
    imprimir_fila('Crank-Nicolson', Ef_cn, Emalla_cn, tiempo_cn, ...
        max(res_cn), sum(iter_cn), max(iter_cn), estado_convergencia(conv_cn));

    %Gráfica errores finales
    figura = figure('Name','Comparacion de errores', 'visible', 'off');
    semilogy([1, 2, 3], [Ef_exp, Ef_imp, Ef_cn], 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8);
    set(gca, 'XTick', 1:3, 'XTickLabel', {'Euler Exp', 'Euler Imp', 'Crank-Nic'});
    xlabel('Método'); ylabel('Error final Ef (escala log)');
    title('Comparación de Errores Finales (Ef)');
    grid on;
    print(figura,'graficas/grafica_errores.png','-dpng');
    close(figura);

    % Gráfica tiempos de ejecución
    figura = figure('Name','Comparacion de tiempos', 'visible', 'off');
    bar([tiempo_exp, tiempo_imp, tiempo_cn]);
    set(gca,'XTick',1:3, 'XTickLabel',{'Euler Exp','Euler Imp','Crank-Nic'});
    ylabel('Tiempo (s)'); title('Tiempos de Ejecución');
    grid on;
    print(figura,'graficas/grafica_tiempos.png','-dpng');
    close(figura);

    %Gráfica perfiles de temperatura t=0,1,5,10
    tiempos_perfil = [0, 1, 5, 10];
    for k = 1:length(tiempos_perfil)
        tiempo_actual = tiempos_perfil(k);
        idx = round(tiempo_actual/dt) + 1;

        figura = figure('Name',sprintf('Perfiles t=%g s',tiempo_actual), 'visible', 'off');
        plot(x, T_exp(idx,:), 'r--o', 'DisplayName','Euler Explícito'); hold on;
        plot(x, T_imp(idx,:), 'b--s', 'DisplayName','Euler Implícito');
        plot(x, T_cn(idx,:), 'g--d', 'DisplayName','Crank-Nicolson');
        plot(x, T_exacta(x,tiempo_actual), 'k-', 'LineWidth', 2, 'DisplayName','Solución Exacta');
        xlabel('Posición x (m)'); ylabel('Temperatura (°C)');
        title(sprintf('Perfiles de temperatura en t = %g s', tiempo_actual));
        legend('Location','northeast'); grid on; hold off;
print(figura,sprintf('graficas/perfil_t_%g.png',tiempo_actual),'-dpng');
        close(figura);
    end
    %Resultados para el resumen final
    resultados.Ef = [Ef_exp, Ef_imp, Ef_cn];
    resultados.Emalla = [Emalla_exp, Emalla_imp, Emalla_cn];
    resultados.tiempo = [tiempo_exp, tiempo_imp, tiempo_cn];
end

function [Ef, Emalla] = calcular_errores(T, x, t, T_exacta)
    T_final_exacta = T_exacta(x,t(end))';
    Ef = max(abs(T(end,:) - T_final_exacta));

    %Error maximo en toda la malla de tiempo
    Emalla = 0;
    for n = 1:length(t)
        T_ref = T_exacta(x,t(n))';
        error_actual = max(abs(T(n,:) - T_ref));
        Emalla = max(Emalla,error_actual);
    end
end

function experimento_refinamiento(L, alpha, u0, T_exacta)
    tf = 10.0;

    %Refinamiento espacial (cambio de M y dt fijo)
    fprintf('\n=== PUNTO 8: REFINAMIENTO ESPACIAL ===\n');
    fprintf('%-10s %-5s %-9s %-9s %-9s %-18s %-12s\n', ...
        'Metodo','M','dx','dt','mu','Ef','Tiempo(s)');

    lista_M = [25,50,100];
    dt_fijo = 0.0025;
    for M = lista_M
        dx = L/M;
        mu = alpha*dt_fijo/dx^2;
        ejecutar_fila_refinamiento('Euler exp.',1,L,alpha,u0,T_exacta,M,dt_fijo,tf,dx,mu);
        ejecutar_fila_refinamiento('Euler imp.',2,L,alpha,u0,T_exacta,M,dt_fijo,tf,dx,mu);
        ejecutar_fila_refinamiento('Crank-Nic.',3,L,alpha,u0,T_exacta,M,dt_fijo,tf,dx,mu);
    end

    %Refinamiento temporal
    fprintf('\n=== PUNTO 8: REFINAMIENTO TEMPORAL ===\n');
    fprintf('%-10s %-5s %-9s %-9s %-9s %-18s %-12s\n', ...
        'Metodo','M','dx','dt','mu','Ef','Tiempo(s)');

    %Refinamiento temporal (cambio dt y M fijo)
    M = 50;
    lista_dt = [0.016,0.008,0.004];
    for dt = lista_dt
        dx = L/M;
        mu = alpha*dt/dx^2;
        ejecutar_fila_refinamiento('Euler exp.',1,L,alpha,u0,T_exacta,M,dt,tf,dx,mu);
        ejecutar_fila_refinamiento('Euler imp.',2,L,alpha,u0,T_exacta,M,dt,tf,dx,mu);
        ejecutar_fila_refinamiento('Crank-Nic.',3,L,alpha,u0,T_exacta,M,dt,tf,dx,mu);
    end
    %Interpretación de lo observado
    fprintf(['Interpretacion: al reducir dx disminuye el error espacial y al reducir dt ' ...
             'disminuye el error temporal. El error total puede dejar de bajar si la ' ...
             'otra fuente de error domina. Los ordenes esperados son O(dt+dx^2) para ' ...
             'Euler y O(dt^2+dx^2) para Crank-Nicolson.\n']);
end

function ejecutar_fila_refinamiento(nombre,metodo,L,alpha,u0,T_exacta,M,dt,tf,dx,mu)
    % Corre el metodo e imprime luego una fila en la tabla
    tic;
    if metodo == 1
        [x,~,U] = euler_explicito(L,alpha,u0,M,dt,tf);
    elseif metodo == 2
        [x,~,U,res,conv,~] = euler_implicito(L,alpha,u0,M,dt,tf);
        if ~conv
            fprintf('ADVERTENCIA: fallo de convergencia en Euler implicito.\n');
        end
    else
        [x,~,U,res,conv,~] = crank_nicolson(L,alpha,u0,M,dt,tf);
        if ~conv
            fprintf('ADVERTENCIA: fallo de convergencia en Crank-Nicolson.\n');
        end
    end
    tiempo = toc;
    T = U + 20;
    Ef = max(abs(T(end,:) - T_exacta(x,tf)'));
    fprintf('%-10s %-5d %-9.5f %-9.4f %-9.4f %-18.6e %-12.4f\n', ...
        nombre,M,dx,dt,mu,Ef,tiempo);
end

function experimento_estabilidad(L,alpha,M)
    fprintf('\n=== PUNTO 9: ESTABILIDAD DE EULER EXPLICITO ===\n');
    tf = 9.6;
    dx = L/M;
    % Se prueba con mu estable(0.4) y mu inestable(0.6)
    for mu = [0.4,0.6]
        dt = mu*dx^2/alpha;
        %Condicion inicial con leve perturbacion
        u0_pert = @(x) (80*sin(pi*x/L) + 1e-6 * ((-1).^(1:length(x)))') .* (x > 0 & x < L);

        [x,t,U] = euler_explicito(L,alpha,u0_pert,M,dt,tf);

        % Desviación máxima respecto a 20 cada tiempo
        max_desviacion = max(abs(U),[],2);

        %Gráficas de estabilidad
        figura = figure('Name',sprintf('Estabilidad mu=%.1f',mu), 'visible', 'off');
        plot(t,max_desviacion,'LineWidth',1.4);
        xlabel('Tiempo t (s)');
        ylabel('max_i |T_i^n - 20| (°C)');
        title(sprintf('Estabilidad de Euler Explícito, mu = %.1f',mu));
        grid on;

        % Si es el caso inestable, limitamos el eje Y para que no explote visualmente
        if mu > 0.5
            ylim([0, 1e6]);
        end

        print(figura,sprintf('graficas/estabilidad_mu_%.1f.png',mu),'-dpng');
        close(figura);
        % Revisión(para saber si explotó)
        idx_explosion = find(max_desviacion > 1e6,1);
        if mu <= 0.5
            fprintf('mu = %.1f (dt=%.5f): condicion teorica estable. ',mu,dt);
        else
            fprintf('mu = %.1f (dt=%.5f): condicion teorica inestable. ',mu,dt);
        end
        if ~isempty(idx_explosion)
            fprintf('Supero 1e6 en t = %.5f s.\n',t(idx_explosion));
        else
            fprintf('Maxima desviacion observada = %.6e.\n',max(max_desviacion));
        end
    end
end

function experimento_enfriamiento(L,alpha,u0,M,dt)
    fprintf('\n=== PUNTOS 10 Y 11: TIEMPO DE ENFRIAMIENTO ===\n');
    tf = 30.0;

    % Se usa Euler implicito por es estable y no tiene restriccion de mu
    [x,t,U,res,conv,iter] = euler_implicito(L,alpha,u0,M,dt,tf);
    if ~conv
        fprintf('ADVERTENCIA: Euler implicito no convergio en todos los pasos.\n');
    end
    T = U + 20;
    max_T_por_tiempo = max(T,[],2);
    %Primer tiempo donde la temperatura maxima baja de 30
    idx_umbral = find(max_T_por_tiempo <= 30,1);

    % Tiempo exacto
    t_exacto = -log(1/8)/(alpha*(pi/L)^2);
    fprintf('Tiempo analitico exacto: %.6f s\n',t_exacto);

    if ~isempty(idx_umbral)
        t_umbral = t(idx_umbral);
        fprintf('Primer tiempo numerico con max(T) <= 30 grados C: %.6f s\n',t_umbral);
        if idx_umbral > 1
            fprintf('Temperatura maxima en el paso anterior: %.6f grados C\n', ...
                max_T_por_tiempo(idx_umbral-1));
        end
        fprintf('Diferencia entre tiempo numerico y exacto: %.6f s\n', ...
            t_umbral-t_exacto);
    else
        fprintf('No se alcanzo 30 grados C antes de tf = %.2f s.\n',tf);
    end
    fprintf('Maximo residuo normalizado: %.3e; iteraciones acumuladas: %d.\n', ...
        max(res),sum(iter));
end

function experimento_mapa_calor(L,alpha,u0,M,dt)
    tf = 10.0;
    [x,t,U] = euler_implicito(L,alpha,u0,M,dt,tf);
    T = U + 20;

    %Mapa de calor de T(x,t)
    figura = figure('Name','Mapa de calor', 'visible', 'off');
    imagesc(t,x,T');
    set(gca,'YDir','normal');
    cb = colorbar();
    ylabel(cb,'Temperatura (°C)');
    xlabel('Tiempo t (s)');
    ylabel('Posición x (m)');
    title('Distribución de Temperatura T(x,t)');
     print(figura,'graficas/mapa_calor.png','-dpng');
    close(figura);

    % Gráfica temperatura del centro
    idx_centro = round(M/2)+1;
    T_centro = T(:,idx_centro);
    figura = figure('Name','Temperatura en el centro', 'visible', 'off');
    plot(t,T_centro,'b-','LineWidth',1.7); hold on;
    plot([t(1),t(end)],[30,30],'r--','LineWidth',1.4);
    xlabel('Tiempo t (s)');
    ylabel('Temperatura en el centro (°C)');
    title('Temperatura en x = L/2');
    legend('Temperatura central','Umbral 30 °C','Location','northeast');
    grid on; hold off;
        print(figura,'graficas/temperatura_centro.png','-dpng');
    close(figura);
end

function imprimir_fila(nombre,Ef,Emalla,tiempo,max_res,iter_acum,max_iter,conv)
    % Imprime una fila de la tabla
    if isnan(max_res)
        fprintf('%-18s | %11.4e | %11.4e | %10.4f | %12s | %10s | %10s | %s\n', ...
            nombre,Ef,Emalla,tiempo,'No aplica','No aplica','No aplica',conv);
    else
        fprintf('%-18s | %11.4e | %11.4e | %10.4f | %12.3e | %10d | %10d | %s\n', ...
            nombre,Ef,Emalla,tiempo,max_res,iter_acum,max_iter,conv);
    end
end

%Estado de convergencia
function estado = estado_convergencia(conv)
    if conv
        estado = 'Si';
    else
        estado = 'No';
    end
end
