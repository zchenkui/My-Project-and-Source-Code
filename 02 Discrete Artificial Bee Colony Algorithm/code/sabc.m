function [result] = sabc(adjMatrix, foodsource, dim, limit, k, maxloop, PrintDetail)

result.looptimes = 0;
result.gbestList = zeros(1, maxloop);
result.gbest     = 0.0;
result.solution  = [];
result.foods     = [];
result.success   = false;
result.evaluations = 0;

foodList = randi([0, 2], foodsource, dim);
fitnessList = zeros(1, foodsource);
for i = 1 : foodsource
    fitnessList(i) = objfunc(adjMatrix, foodList(i, :));
end
[gbest, I] = max(fitnessList);
gbestv = foodList(I, :);
trial = zeros(1, foodsource);

for count = 1 : maxloop
    [foodList, fitnessList, trial, result] = sendEmployedbees(adjMatrix, foodsource, foodList, fitnessList, k, trial, result);
    [probabilityList] = calculateProbability(fitnessList, 1);
    [foodList, fitnessList, trial, result] = sendOnlookerbees(adjMatrix, foodsource, foodList, fitnessList, k, trial, probabilityList, result);
    
    [tmp, I] = max(fitnessList);
    if tmp >= gbest
        gbest = tmp;
        result.gbest = tmp;
        gbestv = foodList(I, :);
    end
    result.gbestList(count) = gbest;
    result.foods = foodList;
    result.looptimes = result.looptimes + 1;
    
    if PrintDetail
        fprintf('\tLoop %5d ===========> %.15e\n', count, gbest);
    end
    
    if gbest == 1.0
        result.solution = gbestv;
        result.success = true;
        break;
    end
    
%     [foodList, fitnessList, trial, result] = sendAllScountBees(adjMatrix, foodList, fitnessList, trial, limit, result);
    [foodList, fitnessList, trial, result] = sendOnlyOneScountBees(adjMatrix, foodList, fitnessList, trial, limit, result);
end

end

function [foodList, fitnessList, trial, result] = sendEmployedbees(adjMatrix, foodsource, foodList, fitnessList, k, trial, result)

[~, dim] = size(foodList);
for i = 1 : foodsource
    neighbor = randi(foodsource);
    while neighbor == i
        neighbor = randi(foodsource);
    end
    
    hd = pdist2(foodList(i, :), foodList(neighbor, :), 'hamming');
    similarity = 1 - rand * hd;
    if rand <= similarity
        optionalSolution = foodList(i, :);
        indices = randsample(dim, k);
        optionalSolution(indices) = foodList(neighbor, indices);
        [optionalFitness] = objfunc(adjMatrix, optionalSolution);
        result.evaluations = result.evaluations + 1;
        
        if optionalFitness >= fitnessList(i)
            foodList(i, :) = optionalSolution;
            fitnessList(i) = optionalFitness;
            trial(i) = 0;
        else
            trial(i) = trial(i) + 1;
        end
    else
        trial(i) = trial(i) + 1;
    end
end

end

function [probabilityList] = calculateProbability(fitnessList, method_to_calculate)
% This function gives the probability of selecting each food source.
% According to the paper, there are two methods, the first one is
% prob(i)=fitness (i)/sum(fitness), and the other is
% prob(i)=a*fitness (i)/max(fitness)+b, where a = 0.9, b = 0.1.

switch method_to_calculate
    case 1
        [maxFitness, ~] = max(fitnessList);
        probabilityList = 0.9 * (fitnessList ./ maxFitness) + 0.1;
        
    case 2
        totalFitness = sum(fitnessList);
        probabilityList = fitnessList ./ totalFitness;
        
    otherwise
        error('No such method \n');
end

end

function [foodList, fitnessList, trial, result] = sendOnlookerbees(adjMatrix, foodsource, foodList, fitnessList, k, trial, probabilityList, result)

t = 1; 
i = 1; 
[~, dim] = size(foodList);
while t <= foodsource
    r = rand;
    if r < probabilityList(i)
        neighbor = randi(foodsource);
        while neighbor == i
            neighbor = randi(foodsource);
        end
        
        hd = pdist2(foodList(i, :), foodList(neighbor, :), 'hamming');
        similarity = 1 - rand * hd;
        
        if rand <= similarity
            optionalSolution = foodList(i, :);
            indices = randsample(dim, k);
            optionalSolution(indices) = foodList(neighbor, indices);
            [optionalFitness] = objfunc(adjMatrix, optionalSolution);
            result.evaluations = result.evaluations + 1;
            
            if optionalFitness >= fitnessList(i)
                foodList(i, :) = optionalSolution;
                fitnessList(i) = optionalFitness;
                trial(i) = 0;
            else
                trial(i) = trial(i) + 1;
            end
        else
            trial(i) = trial(i) + 1;
        end
        
        t = t + 1;
    end
    
    if i == foodsource
        i = 1;
    end
    
    i = i + 1;
end

end

function [foodList, fitnessList, trial, result] = sendAllScountBees(adjMatrix, foodList, fitnessList, trial, limit, result)

trialIndice = find(trial >= limit);
[~, dim] = size(foodList);
foodList(trialIndice, :) = randi([0, 2], length(trialIndice), dim);
for i = 1 : length(trialIndice)
    fitnessList(trialIndice(i)) = objfunc(adjMatrix, foodList(trialIndice(i), :));
    result.evaluations = result.evaluations + 1;
    trial(trialIndice(i)) = 0;
end

end

function [foodList, fitnessList, trial, result] = sendOnlyOneScountBees(adjMatrix, foodList, fitnessList, trial, limit, result)

[~, dim] = size(foodList);
[maxTrial, I] = max(trial);
if maxTrial >= limit
    foodList(I, :) = randi([0, 2], 1, dim);
    fitnessList(I) = objfunc(adjMatrix, foodList(I, :));
    result.evaluations = result.evaluations + 1;
    trial(I) = 0;
end

end