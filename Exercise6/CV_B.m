         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         %% Chantzi Efthymia - Deep Learning - Exercise  6 %%
         %%                     Task B                     %%
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script pertains to the implementation of task B, where deep neural%
% networks are trained and tested on simulated data. The optimal size of %
% the bottleneck node layer is determined by means of a k-fold cross     %
% validation procedure. The simulated data are generated either by the   %
% random-based(F_rand) or image-based(F_image) feedforward network.      %
% When the k-fold cross validation procedure is completed, a plot showing%
% the average root mean sqaured error for each one of the values of H, is%
% produced.                                                              %
% The initialization of the deep neural networks used during the cross   %
% validation and the training of the final networks with the optimal size%
% of bottleneck node layer, is determined by SAE-based initialization.   %
% At the end, plain PCA-based compression, using L latent variables equal%
% to the optimal value for the bottleneck node layer, is performed for   %
% comparison with the respective deep learning.                          %
%                                                                        %
%                                                                        %
% Run this script and a menu will guide you through. More precisely, the % 
% type of feedforward network for data generation(F_rand or F_image),    %
% size of training and test datasets, number of k-folds and the set of   %
% values for the bottleneck node as a row vector are requested.          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;
close all;


fprintf('----------------------------------------- Deep Learning -----------------------------------------\n');
fprintf('   ---------------------------------------- Exercise 6 --------------------------------------\n');
fprintf('     ------------------------------------- Task B ---------------------------------------\n\n');

%% Request used-defined type of network for data generation

networkF = '';
while ((str2double(networkF) ~= 1) && (str2double(networkF) ~= 2))
   
    fprintf('Select type of network:\n');
    fprintf('1. Rand\n');
    fprintf('2. Image\n');
    
    networkF = input('', 's');
    
end
networkF = str2double(networkF);

%% Request used-defined number of training and test examples

if ((networkF == 1) || (networkF == 2))
    
    obs = 0;
    while ((obs <= 0) || (isempty(obs)))

        obs = input('\n Enter the size of the datasets to be generated: \n');

    end
    
end

%% Load dataset into memory
   
architecture = [50 90 1600];
if (networkF == 1)
    
    % generate training dataset 
    fprintf('----> Training Dataset <----');
    [train_dataset, train_inputs] = generateRandData_taskB(0.1, architecture, 'logsig', obs, 10^-3);
     
    % generate test dataset
    fprintf('----> Test Dataset <----');
    [test_dataset, test_inputs] = generateRandData_taskB(0.1, architecture, 'logsig', obs, 10^-3);
     
    strFigure = 'F_{rand}';
    
    
else
    
    cd data
    
    load 'deepnet_90_50_90.mat';
    load 'data.mat';
    
    cd ..
    
    % generate training dataset 
    [train_dataset, train_inputs] = generateImageData_taskB(deepnet, data, architecture, 'logsig', obs, 10^-8);
    
    % generate test dataset
    [test_dataset, test_inputs] = generateImageData_taskB(deepnet, data, architecture, 'logsig', obs, 10^-8);
    
    strFigure = 'F_{image}';
    
end


%% Request user-defined parameters for the CV procedure

K = 0;
while ((sum(mod(K, 1)) > 0) || (sum(K < 1) > 0) || (isempty(K)))

    K = input('\n Enter the number of CV folds: \n');
    
end

%% Request used-defined row vector of set for hidden layer size

H = 0;
while ((sum(mod(H, 1)) > 0) || (sum(H < 1) > 0) || (isempty(H)))

    H = input('\n Enter the bottleneck sizes as a row vector: \n');
    
end
len = length(H);

%% additional variables

% initialization of CV inner error estimate
E = zeros(K, len);
RE = zeros(K, len);

time_start = tic;

%% CV loop

% indices of CV partitions
ind = crossvalind('Kfold', obs, K);

for i = 1 : K

    testing_ind = (ind == i);        % indices for the K fold used for evaluation
    training_ind = ~testing_ind;     % indices for the (K - 1) folds used for training

    train_images = train_inputs(:, training_ind);
    test_images = train_inputs(:, testing_ind);

    fprintf('\n  ------------------->  CV: Iter %d of %d  <------------------- \n', i, K);

    for h = 1 : len

        fprintf('H = %d ...\n', H(h));
        fprintf('\n');

        architectureVector = [architecture(3) architecture(2) H(h) architecture(2) architecture(3)];
        
        net = SAEInit(architectureVector, train_images, 400);
               
        %% Training

        net = train(net, train_images, train_images);

        %% Testing on the Kth inner fold

        outputs_test_images = net(test_images);
        [MSE, RMSE] = estimateTotalMSE(test_images, outputs_test_images, 'im', 'rmse');
        E(i, h) = MSE;
        RE(i, h) = RMSE;

        
    end

end
time_end = toc(time_start);

% time required for the CV loop
t = datevec(time_end./(60*60*24));
fprintf('\n Time elapsed for %d-fold CV: %.2f (hrs) %.2f (min) %.2f (sec)\n', K, t(4), t(5), t(6));

E_avg = mean(E); 
[r, c] = find(E_avg(1, :) == min(E_avg(1, :)));
E_avg_star = min(E_avg);
   
RE_avg = mean(RE);
[rR, cR] = find(RE_avg(1, :) == min(RE_avg(1, :)));
RE_avg_star = min(RE_avg);


%% RMSE plot of Cross Validation

figure();
plot(H, RE_avg, 'c*-', 'linewidth', 1.5);
hold on;
scatter(H(cR), RE_avg(cR), 'm*', 'linewidth', 1.5);
hold off;
xlim([(min(H) - 10) (max(H) + 10)]);
ylim([(min(RE_avg) - min(RE_avg)/10^4) (max(RE_avg) + max(RE_avg)/10^4)]);
%ylim([(min(E_avg)) (max(E_avg))]);  % for random initialization
xlabel('H', 'fontweight', 'bold');
ylabel('RMSE_{avg}', 'fontweight', 'bold');
grid on;
title(sprintf('Average RMSE from %d-fold Cross Validation\n %s data generation of %d examples\n\n', K, strFigure, obs), 'fontweight', 'bold');

%% Train Deep Network using optimal H from CV

architectureVector(3) = H(cR);

net = SAEInit(architectureVector, train_inputs, 400);

net = train(net, train_inputs, train_inputs);

%% Test trained network on the seen training dataset using optimal H from CV

outputs_train = net(train_inputs);
[totalMSE_training, totalRMSE_training] = estimateTotalMSE(train_inputs, outputs_train, 'im', 'rmse');
fprintf('\n');
fprintf('-------------------- Training Performance --------------------\n');
fprintf('Total MSE on the training dataset: %.5f\n', totalMSE_training);
fprintf('Total RMSE on the training dataset: %.5f\n', totalRMSE_training);

% absolute difference between original and predicted values of training
% dataset
dif_train = abs(train_inputs - outputs_train);

%% Test trained network on the unseen test dataset

outputs_test = net(test_inputs);
[totalMSE_test, totalRMSE_test] = estimateTotalMSE(test_inputs, outputs_test, 'im', 'rmse');
fprintf('\n');
fprintf('-------------------- Testing Performance --------------------\n');
fprintf('Total MSE on the test dataset: %.5f\n', totalMSE_test);
fprintf('Total RMSE on the test dataset: %.5f\n', totalRMSE_test);

% absolute difference between original and predicted values of test
% dataset
dif_test = abs(test_inputs - outputs_test);

%% Plot of original vs. predicted values on training and test datasets using the first 5 examples

figure();
subplot(1, 2, 1);
hold on;
for i = 1 : 5
    
    plot(train_inputs(:, i), 'm.');
    plot(outputs_train(:, i), 'go');
    
end
hold off;
grid on;
xlabel('dimensions', 'fontweight', 'bold');
ylim([-0.1 1.1]);
ylabel('values', 'fontweight', 'bold');
title(sprintf('Train Dataset (RMSE = %.5f)', totalRMSE_training), 'fontweight', 'bold');
legend('Real','Predicted','Location','Best');

subplot(1, 2, 2);
hold on;
for i = 1 : 5
    
    plot(test_inputs(:, i), 'm.');
    plot(outputs_test(:, i), 'go');
    
end
hold off;
grid on;
ylim([-0.1 1.1]);
xlabel('dimensions', 'fontweight', 'bold');
ylim([-0.1 1.1]);
ylabel('values', 'fontweight', 'bold');
title(sprintf('Test Dataset (RMSE = %.5f)', totalRMSE_test), 'fontweight', 'bold');
legend('Real','Predicted','Location','Best');
    
hold on;
axes;
h = title(sprintf('Original vs. Predicted Values of the first 5 out of %d %s generated examples\n {%d-%d-%d-%d-%d}\n\n\n', obs, strFigure, architectureVector(1), architectureVector(2), architectureVector(3), architectureVector(4), architectureVector(5)), 'fontweight', 'bold');
set(gca, 'Visible', 'off');
set(h, 'Visible', 'on');
hold off; 

%% Plot of mean absolute difference between the original and predicted outputs on training and test datasets 

figure();
subplot(1, 2, 1);
hold on;
for i = 1 : obs
    
    plot(i, mean(dif_train(:, i)), 'ms', 'MarkerSize', 10);
    
end
hold off;
grid on;
title('Train Dataset', 'fontweight', 'bold');
xlabel('examples', 'fontweight', 'bold');
ylabel('mean absolute difference', 'fontweight', 'bold');


subplot(1, 2, 2);
hold on;
for i = 1 : obs
    
    plot(i, mean(dif_test(:, i)), 'gs', 'MarkerSize', 10);
    
end
hold off;
grid on;
title('Test Dataset', 'fontweight', 'bold');
xlabel('examples', 'fontweight', 'bold');
ylabel('mean absolute difference', 'fontweight', 'bold');

hold on;
axes;
h = title(sprintf('Mean absolute difference between the original and predicted %d %s generated examples\n {%d-%d-%d-%d-%d}\n\n\n', obs, strFigure, architectureVector(1), architectureVector(2), architectureVector(3), architectureVector(4), architectureVector(5)), 'fontweight', 'bold');
set(gca, 'Visible', 'off');
set(h, 'Visible', 'on');
hold off; 

%% PCA and reconstruction errors on training and test sets
fprintf('\n ------------------------------------ PCA compression -------------------------------------\n');

%% training dataset

% L = H_optimal gained by 2-fold CV
[coeff_train_Hopt, score_train_Hopt, latent_train_Hopt, mu_train_Hopt, totalVarPCs_M_train_Hopt, reconstructions_train_Hopt, totalMSE_PCA_train_Hopt, totalRMSE_PCA_train_Hopt] = PCAonData(train_inputs, H(c), 'im', 'rmse');
fprintf('\n');
fprintf('Total MSE after compression to %d dimensions on the training dataset: %.15f\n', H(cR), totalMSE_PCA_train_Hopt);
fprintf('Total RMSE after compression to %d dimensions on the training dataset: %.15f\n', H(cR), totalRMSE_PCA_train_Hopt);

if (H(cR) ~= H(1))
   
    % PCA on training set using L = H_min
    [coeff_train_Hmin, score_train_Hmin, latent_train_Hmin, mu_train_Hmin, totalVarPCs_M_train_Hmin, reconstructions_train_Hmin, totalMSE_PCA_train_Hmin, totalRMSE_PCA_train_Hmin] = PCAonData(train_inputs, H(1), 'im', 'rmse');
    fprintf('\n');
    fprintf('Total MSE after compression to %d dimensions on the training dataset: %.15f\n', H(1), totalMSE_PCA_train_Hmin);
    fprintf('Total RMSE after compression to %d dimensions on the training dataset: %.15f\n', H(1), totalRMSE_PCA_train_Hmin);
    
end

%% test dataset

% L = H_optimal gained by k-fold CV
[coeff_test_Hopt, score_test_Hopt, latent_test_Hopt, mu_test_Hopt, totalVarPCs_M_test_Hopt, reconstructions_test_Hopt, totalMSE_PCA_test_Hopt, totalRMSE_PCA_test_Hopt] = PCAonData(test_inputs, H(c), 'im', 'rmse');
fprintf('\n');
fprintf('Total MSE after compression to %d dimensions on the test dataset: %.15f\n', H(cR), totalMSE_PCA_test_Hopt);
fprintf('Total RMSE after compression to %d dimensions on the test dataset: %.15f\n', H(cR), totalRMSE_PCA_test_Hopt);

if (H(cR) ~= H(1))
   
    % PCA on training set using L = H_min
    [coeff_test_Hmin, score_test_Hmin, latent_test_Hmin, mu_test_Hmin, totalVarPCs_M_test_Hmin, reconstructions_test_Hmin, totalMSE_PCA_test_Hmin, totalRMSE_PCA_test_Hmin] = PCAonData(test_inputs, H(1), 'im', 'rmse');
    fprintf('\n');
    fprintf('Total MSE after compression to %d dimensions on the test dataset: %.15f\n', H(1), totalMSE_PCA_test_Hmin);
    fprintf('Total RMSE after compression to %d dimensions on the test dataset: %.15f\n', H(1), totalRMSE_PCA_test_Hmin);
    
end
