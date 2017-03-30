clear; clc;

% kList = [1, 2, 3, 4, 5];
% limitList = [110, 120];
k = 2;
limit = 90;
gnum = 100;
n = 90;
dList = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
maxloop = 10000;
foodnumber = 200;
PrintDetail = false;

srlist = zeros(1, length(dList));
evlist = zeros(1, length(dList));
for i = 1 : length(dList)
    d = dList(i);
    sr = 0; 
    ev = 0;
    fprintf('d = %f begin\n', d);
    for j = 1 : gnum
        [gr, adjMatrix] = generateGraph(n, d);
        [result] = sabc(adjMatrix, foodnumber, n, limit, k, maxloop, PrintDetail);
        if result.success
            fprintf('graph %3d: d = %f ======> success | evaluation: %.5e\n', j, d, result.evaluations);
            sr = sr + 1;
        else
            fprintf('graph %3d: d = %f ======> fail | evaluation: %.5e\n', j, d, result.evaluations);
        end
        ev = ev + result.evaluations;
    end
    fprintf('d = %f, all finished | success %d times | average evaluations %.5e\n\n', d, sr, ev / gnum);
    srlist(i) = sr;
    evlist(i) = ev / gnum;
end

save('test90_15_40.mat');