function lelisFuzzyC_Means()

    clear
    close all
    clc

    % Le, exibe e pega as dimensoes da imagem
    img = imread('img02.jpg');
    imshow(img);
    [x,y,z] = size(img);

    % De imagem a matriz de valores empilhados
    newSize = [x*y 3];
    stackedPixels = reshape(img, newSize);

    % Carrega o conjunto de dados (X -> m*d)
    X = double(stackedPixels);

    % Carrega o conjunto de dados (X -> m*d)
    %X = load('fcm_dataset.mat');
    %X = X.x;

    % Numero maximo de iteracoes
    maxIterations = 20;

    % n = numero de observacoes/elementos/pontos/padroes
    % d = numero de dimensoes (pto = 2 / img = 3 (RGB))
    [n d] = size(X);

    % k = numero de grupos/centros/agrupamentos/clusters
    k = 4;

    % m = constante	que	determina a influencia dos pesos
    m = 2;

    % Criando a matriz de particao
    U = createPartMtx(n,k);

    % O algoritmo roda ate atingir o criterio de parada (ou max de iteracoes)
    for r=1:maxIterations

        % Cria a mtx de centros e atualiza a mtx de particao
        C = createCenterMtx(X,U,k,d,n,m);
        U = updatePartMtx(X,C,k,n,m);

        % Verifica a condicao de parada
        costValue(r) = costFunction(X,C,U,m,n,k);
        if r ~= 1
            if abs(costValue(r) - costValue(r-1)) < 1e-5
                break;
            end
        end

    end

    % Plota os graficos caso a entrada seja um dataset
    %plotGraphs(X,C,U,k);

    % Plota a img resultante caso a entrada seja uma img
    plotImg(X,C,U,n,x,y,z);

end

function U = createPartMtx(n,k)

    % Matriz de particao inicial normalizada (U -> n*k e 0 < U_ij < 1)
    U = rand (n,k);
    for i =1:n
        sum = 0;
        for j = 1:k
           sum = sum+U(i,j);
        end
        U(i,:) = U(i,:)/sum;
    end
    
end

function C = createCenterMtx(X,U,k,d,n,m)
    
    % Preenche a mtx de centros inicial com zeros (numerador+denominador)
    cNum = zeros (k,d);
    cDen = zeros (k,d);

    % Obtem a matriz de Centros (C -> k*d)
    for j=1:k
        for i=1:n
            cNum(j,:) = (U(i,j)^m)*X(i,:) + cNum(j,:);
            cDen(j,:) = (U(i,j)^m) + cDen(j,:);
        end
    end

    C = cNum./cDen;
    
end

function U = updatePartMtx(X,C,k,n,m)

    % Preenche a nova matriz de particao com zeros (numerador+denominador)
    uNum = 0;
    uDen = 0;

    % Obtem a nova matriz de particao (U -> m*k)
    U = zeros(n,k);
    for j=1:k
        for i=1:n
            if (pdist2(X(i,:),C(j,:)) == 0)
                U(i,:) = 0;
                U(i,j) = 1;
                break
            else
                uNum = pdist2(X(i,:),C(j,:));
                aux = 0;
                for l=1:k
                   uDen = pdist2(X(i,:),C(l,:));
                   aux = aux + ((uNum/uDen)^(2/(m-1)));
                end
            end
            U(i,j) = 1/aux;
        end
    end
    
end

function plotGraphs(X,C,U,k)

    % Classifica cada ponto no cluster com o maior valor
    U = U';    
    maxU = max(U);

    % Plota os dados agrupados(clustered) e os centros
    figure(1)

    % Plota ate 7 cores, ou seja -> 7 clusters/centros/agrupamentos/grupos
    colorstring = 'brgymck';

    for l=1:k
        colorPlot = strcat('o',colorstring(l));
        index = find(U(l,:) == maxU);

        plot(X(index,1),X(index,2),colorPlot)
        hold on
        plot(C(l,1),C(l,2),'xk','MarkerSize',15,'LineWidth',3)
    end

    axis([2 4.5 3 5.5])
    hold off
    
    % PARA TESTE
    % Testando pela funcao fcm
    [C,U] = fcm(X,k);
    maxU = max(U);
    figure(2)
    for l =1:k
        colorPlot = strcat('o',colorstring(l));
        index = find(U(l,:) == maxU);

        plot(X(index,1),X(index,2),colorPlot)
        hold on
        plot(C(l,1),C(l,2),colorPlot,'MarkerSize',15,'LineWidth',3)
    end
    axis([2 4.5 3 5.5])
    hold off
    % PARA TESTE

end

function plotImg(X,C,U,n,x,y,z)

    % Pega o valor max de cada uma das linhas da mtx U e o respectivo indice
    [maxValues, maxIndexes] = max(U, [ ], 2);
    
    % Cada pixel da mtx X(entrada) recebe o valor do cluster mais relevante
    for i=1:n
        X(i,:) = int64(C(maxIndexes(i),:));
    end
    
    % Reconstruindo a imagem para plota-la
    img_rebuilt = reshape(X, [x, y, z]); 
    img_rebuilt = uint8(img_rebuilt);
    figure();
    imshow(img_rebuilt);

end

function costValue = costFunction(X,C,U,m,n,k)
    
    % Calcula a funcao objetivo (ou funcao de custo) a ser minimizada
    costValue = 0;
    for j = 1:k
        for i = 1:n
            costValue = costValue + (U(i,j)^m)*(pdist2(X(i,:),(C(j,:)))^2);
        end
    end
    
end