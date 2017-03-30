    function [fitness] = objfunc(adjMatrix, sol)

upperAdjMatrix = triu(adjMatrix);
m = sum(sum(upperAdjMatrix)); % number of edges
[rs, cs] = find(upperAdjMatrix == 1);
conflicts = sum(sol(rs) == sol(cs));

fitness = 1.0 - conflicts / m;

end