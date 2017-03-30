function [gr, adjMatrix] = generateGraph(n, d)

m = floor(n * d); % m is the edges number.
adjMatrix = zeros(n, n);
part1 = randi([0, 1], floor(n/3));
part2 = randi([0, 1], floor(n/3));
part3 = randi([0, 1], floor(n/3));

links = sum(sum(part1)) + sum(sum(part2)) + sum(sum(part3));
while links ~= m
    rnd1 = randi([1, 3], 1);
    rnd2 = randi([1, floor(n/3)], 1);
    rnd3 = randi([1, floor(n/3)], 1);
    
    switch rnd1
        case 1
            if links > m && part1(rnd2, rnd3) == 1
                part1(rnd2, rnd3) = 0;
                links = links - 1;
            elseif links < m && part1(rnd2, rnd3) == 0
                part1(rnd2, rnd3) = 1;
                links = links + 1;
            end
            
        case 2
            if links > m && part2(rnd2, rnd3) == 1
                part2(rnd2, rnd3) = 0;
                links = links - 1;
            elseif links < m && part2(rnd2, rnd3) == 0
                part2(rnd2, rnd3) = 1;
                links = links + 1;
            end
             
        case 3
            if links > m && part3(rnd2, rnd3) == 1
                part3(rnd2, rnd3) = 0;
                links = links - 1;
            elseif links < m && part3(rnd2, rnd3) == 0
                part3(rnd2, rnd3) = 1;
                links = links + 1;
            end
    end
end

adjMatrix(1 : floor(n/3), floor(n/3) + 1 : 2 * floor(n/3)) = part1;
adjMatrix(1 : floor(n/3), 2 * floor(n/3) + 1 : n) = part2;
adjMatrix(floor(n/3) + 1 : 2 * floor(n/3), 2 * floor(n/3) + 1 : n) = part3;

adjMatrix = adjMatrix + adjMatrix';

gr = graph(adjMatrix);

end