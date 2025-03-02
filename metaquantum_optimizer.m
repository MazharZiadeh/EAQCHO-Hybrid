%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Self-Adaptive Quantum-Classical Evolutionary Optimization
% with Meta-Evolutionary Controllers + Visualizations
% Objective: Minimize the Sphere function f(x) = sum(x.^2) below 1e-6
% Date: March 2, 2025 
% Author: Mazhar Ziadeh + Grok3 :)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc; close all;

%% --------------------- Problem & Algorithm Parameters ---------------------
dim = 10;                     % Problem dimension
popSize = 100;                % Population size
metaPopSize = 10;             % Meta-controller population size
maxGen = 2000;                % Max generations
bounds = [-5, 5];             % Variable bounds
qThetaInit = 0.01*pi;         % Initial quantum rotation angle
qThetaMin = 0.0005*pi;        % Minimum quantum rotation angle
stagnationThreshold = 100;    % Gens w/o improvement => re-init
reinitFraction = 0.02;        % Reinit fraction of pop if stagnant

%% ------------------ Initialize Base Population (Quantum-Classical) -------
population = zeros(popSize, dim);
qProb = 0.5 * ones(popSize, dim);
adaptParams = rand(popSize, 2);
fitness = zeros(popSize, 1);

for i = 1:popSize
    population(i,:) = bounds(1) + (bounds(2)-bounds(1)) * rand(1, dim);
    fitness(i) = sphereFunc(population(i,:));
end

%% ------------------ Initialize Meta-Evolutionary Population -------------
metaPopulation = rand(metaPopSize,3);  % [qw, cw, ar] in [0,1]
metaFitness = zeros(metaPopSize,1);

%% ------------------ Tracking Arrays & Variables --------------------------
bestFitnessHistory = zeros(maxGen,1);
avgFitnessHistory  = zeros(maxGen,1);

% Track meta-parameter means
meanMetaQW = zeros(maxGen,1);
meanMetaCW = zeros(maxGen,1);
meanMetaAR = zeros(maxGen,1);

% Track re-initialization events (for plotting)
reinitEvents = [];

lastImprovementGen = 0;

%% ------------------ Main Evolutionary Loop ------------------------------
for gen = 1:maxGen
    
    % Adaptive quantum rotation angle with possible "stagnation boost"
    qThetaBase = qThetaInit * exp(-2*gen / maxGen) + qThetaMin;
    
    % If no improvement in last ~50 gens, boost rotation angle by 50%
    if gen > 50 && bestFitnessHistory(max(1,gen-1)) >= ...
                   bestFitnessHistory(max(1,gen-50))
        qTheta = qThetaBase * 1.5;
    else
        qTheta = qThetaBase;
    end
    
    %% Step 1: Evaluate each meta-controller on a subset
    subsetSize = floor(popSize / metaPopSize);
    for m = 1:metaPopSize
        subsetIdx = (m-1)*subsetSize + 1 : min(m*subsetSize, popSize);
        
        subPop    = population(subsetIdx,:);
        subQProb  = qProb(subsetIdx,:);
        subAdapt  = adaptParams(subsetIdx,:);
        subFit    = fitness(subsetIdx);
        
        % Evolve the subset
        [newSubPop, newSubQProb, newSubAdapt, deltaFitness] = evolveSubset(...
            subPop, subQProb, subAdapt, subFit, metaPopulation(m,:), bounds, qTheta);
        
        population(subsetIdx,:) = newSubPop;
        qProb(subsetIdx,:)      = newSubQProb;
        adaptParams(subsetIdx,:) = newSubAdapt;
        fitness(subsetIdx)      = sphereFunc(newSubPop);
        
        metaFitness(m) = mean(deltaFitness);
    end
    
    %% Step 2: Evolve Meta-Controllers
    [metaPopulation, metaFitness] = evolveMetaPopulation(metaPopulation, metaFitness);
    
    % Record the mean meta-parameters
    meanMetaQW(gen) = mean(metaPopulation(:,1));
    meanMetaCW(gen) = mean(metaPopulation(:,2));
    meanMetaAR(gen) = mean(metaPopulation(:,3));
    
    %% Step 3: Evaluate Base Population again (though we already did)
    for i = 1:popSize
        fitness(i) = sphereFunc(population(i,:));
    end
    
    %% Track best & average fitness
    bestFitness = min(fitness);
    avgFitness  = mean(fitness);
    
    bestFitnessHistory(gen) = bestFitness;
    avgFitnessHistory(gen)  = avgFitness;
    
    %% Check for long-term stagnation => Reinit fraction of pop
    if gen > stagnationThreshold
        if bestFitnessHistory(gen) >= bestFitnessHistory(gen - stagnationThreshold)
            reinitCount = floor(reinitFraction * popSize); 
            reinitIdx   = randperm(popSize, reinitCount);
            for ri = reinitIdx
                population(ri,:) = bounds(1) + (bounds(2)-bounds(1)) * rand(1,dim);
                qProb(ri,:)      = 0.5 * ones(1, dim);
                fitness(ri)      = sphereFunc(population(ri,:));
            end
            fprintf('Re-initialized %d individuals at Generation %d due to stagnation\n',...
                reinitCount, gen);
            reinitEvents = [reinitEvents; gen]; %#ok<AGROW>
        end
    end
    
    % Update "last improvement" tracker
    if gen>1 && (bestFitness < bestFitnessHistory(gen-1))
        lastImprovementGen = gen;
    end
    
    %% Console Progress
    fprintf('Gen %d: Best=%.6f, Avg=%.6f, qTheta=%.6f, MetaQW=%.3f, CW=%.3f, AR=%.3f\n',...
        gen, bestFitness, avgFitness, qTheta,...
        meanMetaQW(gen), meanMetaCW(gen), meanMetaAR(gen));
    
    %% Early stopping
    if bestFitness < 1e-6
        fprintf('Converged at Generation %d\n', gen);
        break;
    end
end

%% ------------------ Final Results & Output -------------------------------
[bestFitness, bestIdx] = min(fitness);
bestSolution = population(bestIdx,:);
fprintf('Best Solution:\n');
disp(bestSolution);
fprintf('Best Fitness: %.6f\n', bestFitness);

%% ------------------ PLOTTING --------------------------------------------
% 1) Best + Average Fitness
figure('Name','Fitness Curves','Color',[1 1 1]);
hold on; grid on;
numGens = find(bestFitnessHistory,1,'last'); % actual # generations used
if isempty(numGens), numGens = maxGen; end

plot(1:numGens, bestFitnessHistory(1:numGens), 'r-', 'LineWidth',1.5);
plot(1:numGens, avgFitnessHistory(1:numGens),  'b-', 'LineWidth',1.5);

% Mark re-init events on best-fitness line
if ~isempty(reinitEvents)
    scatter(reinitEvents, bestFitnessHistory(reinitEvents), 40, 'k', 'filled');
    legend('Best Fitness','Average Fitness','Re-inits','Location','Best');
else
    legend('Best Fitness','Average Fitness','Location','Best');
end
xlabel('Generation');
ylabel('Fitness');
title('Fitness Progress + Re-init Events');

% 2) Meta-Parameter Means
figure('Name','Meta-Param Means','Color',[1 1 1]);
subplot(3,1,1); hold on; grid on;
plot(1:numGens, meanMetaQW(1:numGens), 'LineWidth',1.5, 'Color',[0.85 0.33 0.10]);
ylabel('Mean QW'); title('Meta-Parameter Averages');
subplot(3,1,2); hold on; grid on;
plot(1:numGens, meanMetaCW(1:numGens), 'LineWidth',1.5, 'Color',[0.49 0.18 0.56]);
ylabel('Mean CW');
subplot(3,1,3); hold on; grid on;
plot(1:numGens, meanMetaAR(1:numGens), 'LineWidth',1.5, 'Color',[0.47 0.67 0.19]);
ylabel('Mean AR'); xlabel('Generation');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           HELPER FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f = sphereFunc(x)
    % Minimization objective: sum of squares
    f = sum(x.^2, 2);
end

function [newPop, newQProb, newAdapt, deltaFitness] = evolveSubset(pop, qProb, adaptParams,...
    fitness, metaParams, bounds, qTheta)

    popSize = size(pop,1);
    dim = size(pop,2);
    
    % Enforce minimum QW=0.5, min CW=0.5
    qw = max(0.5, metaParams(1));
    cw = max(0.5, metaParams(2));
    ar = metaParams(3);
    
    newPop   = pop;
    newQProb = qProb;
    newAdapt = adaptParams;
    deltaFitness = zeros(popSize,1);
    
    for i=1:popSize
        oldFit = fitness(i);
        
        % Tournament selection
        parentIdx = tournamentSelect(fitness);
        parent    = pop(parentIdx,:);
        
        % Quantum step
        if rand < qw
            deltaQ = qTheta * (2*rand(1,dim) - 1);
            newQ   = qProb(i,:) + deltaQ;
            newQProb(i,:) = min(1, max(0, newQ));
            % Reconstruct real solution
            newPop(i,:)   = bounds(1) + (bounds(2)-bounds(1)) .* newQProb(i,:);
        end
        
        % Classical step
        if rand < cw
            mutationRate = adaptParams(i,1);
            % 20% chance of Cauchy
            if rand<0.2
                if oldFit < 0.0001
                    % dynamic scale for very small fitness
                    mutant = parent + 5 * mutationRate * tan(pi*(rand(1,dim)-0.5));
                else
                    mutant = parent + 2 * mutationRate * tan(pi*(rand(1,dim)-0.5));
                end
            else
                mutant = parent + 0.5 * mutationRate * randn(1,dim);
            end
            % Bound check
            mutant = min(bounds(2), max(bounds(1), mutant));
            
            % Crossover
            crossProb = adaptParams(i,2);
            crossMask = (rand(1,dim) < crossProb);
            newPop(i,crossMask) = mutant(crossMask);
        end
        
        % Evaluate new solution
        newFit = sphereFunc(newPop(i,:));
        improvement = oldFit - newFit;
        if newFit > oldFit
            % revert
            newPop(i,:) = pop(i,:);
            newQProb(i,:) = qProb(i,:);
            improvement = 0;
            newFit = oldFit;
        end
        deltaFitness(i) = improvement;
        
        % Self-adapt
        newParams = adaptParams(i,:) + ar*(rand(1,2) - 0.5);
        newAdapt(i,:) = min(1, max(0, newParams));
    end
end

function [newMetaPop, newMetaFitness] = evolveMetaPopulation(metaPop, metaFitness)
    metaPopSize = size(metaPop,1);
    dim = size(metaPop,2);
    newMetaPop = metaPop;
    newMetaFitness = metaFitness;
    
    for i = 1:metaPopSize
        p1 = tournamentSelect(-metaFitness); % negative => maximize
        p2 = tournamentSelect(-metaFitness);
        
        % crossover
        if rand < 0.8
            alpha = rand;
            newMetaPop(i,:) = alpha * metaPop(p1,:) + (1-alpha)*metaPop(p2,:);
        end
        
        % mutation
        if rand < 0.1
            newMetaPop(i,:) = newMetaPop(i,:) + 0.1 * randn(1,dim);
        end
        
        % Bound clamp
        newMetaPop(i,:) = min(1, max(0, newMetaPop(i,:)));
        
        % min QW=0.5, min CW=0.5
        newMetaPop(i,1) = max(0.5, newMetaPop(i,1));
        newMetaPop(i,2) = max(0.5, newMetaPop(i,2));
    end
end

function idx = tournamentSelect(fitnessVec)
    % Minimizing fitness => pick the smaller
    tSize = 2;
    cands = randi(length(fitnessVec), [1, tSize]);
    [~, bestLocal] = min(fitnessVec(cands));
    idx = cands(bestLocal);
end
