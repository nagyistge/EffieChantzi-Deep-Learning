         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         %% Chantzi Efthymia - Deep Learning - Exercise  6 %%
         %%                     Task A                     %%
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script pertains to the implementation of task A, where the        %
% prediction modeling is performed on the time-lapse microscopy movies.  %
% The current implementation involves the prediction modeling on two     %
% different movies; movie 18(Large interphase nuclei) and movie28        %
% (Fragmented nuclei), which are taken from the second set of the        %
% supplementary movies. The reasons why only these two movies have been  %
% selected are explained thoroughly in the report.                       %
% The prediction modeling is performed by feedforward neural networks    %
% with either 1 or 2 hidden layers. The optimal size of the bottleneck   %
% node layer is determined by means of a k-fold cross validation         %
% procedure. When this procedure is completed, a plot illustrating the   %
% average error estimates is produced.                                   %
% Furthermore, the initialization of weights and biases of these networks%
% is done either by PCA-based or SAE-based initialization. At the end, a %
% plot of five original versus the respective reconstructed images is    %
% obtained with the total mean squared error on the whole test dataset.  %
%                                                                        %
%                                                                        %
% Run this script and a menu will guide you through. More precisely, the % 
% movie(either 18 or 28) for the prediction modeling, the number of the  %
% L latent variables used for the PCA-based compression, the number of   % 
% k-folds, the number of hidden layers, set of values for the bottleneck %
% node as a row vector and the initialization type(either PCA or SAE)    %
% are requested from the user.                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;
close all;

fprintf('----------------------------------------- Deep Learning -----------------------------------------\n');
fprintf('   ---------------------------------------- Exercise 6 --------------------------------------\n');
fprintf('     ------------------------------------- Task A ---------------------------------------\n\n');

%% Request used-defined movie and load training and test datasets into memory

movie = '';
while ((str2double(movie) ~= 1) && (str2double(movie) ~= 2))
   
    fprintf('Select time-lapse microscopy movie:\n');
    fprintf('1. Large interphase nuclei (movie 18)\n');
    fprintf('2. Fragmented nuclei (movie 28)\n');
    
    movie = input('', 's');
    
end
movie = str2double(movie);

if (movie == 1)
   
    cd data
    
    % training dataset
    load 'train_images_45_m18.mat';
    load 'train_predictions_45_m18.mat';
    
    % test dataset
    load 'test_images_45_m18.mat';
    load 'test_predictions_45_m18.mat';

    cd ..
    
    train_images = train_images_45_m18;
    train_predictions = train_predictions_45_m18;
    
    test_images = test_images_45_m18;
    test_predictions = test_predictions_45_m18;
    
else
    
    cd data
    
    % training dataset
    load 'train_images_45_m28.mat';
    load 'train_predictions_45_m28.mat';
    
    % test dataset
    load 'test_images_45_m28.mat';
    load 'test_predictions_45_m28.mat';
    
    cd ..
    
    train_images = train_images_45_m28;
    train_predictions = train_predictions_45_m28;
    
    test_images = test_images_45_m28;
    test_predictions = test_predictions_45_m28;
    
end
    
reshapeVector = [45 45];
mode = 'im';
   
%% Request user-defined reduced dimensions

% principal components of the M reduced dimensions
PCs_M = '';
while ((str2double(PCs_M) <= 0) || (mod(str2double(PCs_M), 1) ~= 0))

    PCs_M = input('\n Enter a positive value for the L latent variables: \n', 's');

end
PCs_M = str2double(PCs_M);

%% PCA on training data
[coeff_inputs, score_inputs, latent_inputs, mu_inputs, totalVarPCs_M_inputs, reconstructions_inputs, totalMSE_inputs] = PCAonData(train_images, PCs_M, mode);

% compressed inputs
inputs = score_inputs(:, (1 : PCs_M))';


[coeff_targets, score_targets, latent_targets, mu_targets, totalVarPCs_M_targets, reconstructions_targets, totalMSE_targets] = PCAonData(train_predictions, PCs_M/2, mode);

% compressed targets
targets = score_targets(:, (1 : PCs_M/2))';


[dim, obs] = size(inputs);

%% Request user-defined parameters for the CV procedure
K = 0;
while ((sum(mod(K, 1)) > 0) || (sum(K < 1) > 0) || (isempty(K)))

    K = input('\n Enter the number of CV folds: \n');
    
end


%% Request used-defined number of hidden layers (either 1 or 2)

numOfHL = '';
while ((str2double(numOfHL) ~= 1) && (str2double(numOfHL) ~= 2))
   
    fprintf('\n Select number of hidden layers:\n');
    fprintf('1. One\n');
    fprintf('2. Two\n');
    
    numOfHL = input('', 's');
    
end
numOfHL = str2double(numOfHL);

if (numOfHL == 1)
   
    architectureVector = [dim 0 dim/2];
    
else
    
    architectureVector = [dim dim 0 dim/2];
    
end

%% Request used-defined row vector of set for hidden layer size

H = 0;
while ((sum(mod(H, 1)) > 0) || (sum(H < 1) > 0) || (isempty(H)))

    H = input('\n Enter the bottleneck sizes as a row vector: \n');
    
end
len = length(H);

%% Request user-defined initialization method

initType = '';
while ((str2double(initType) ~= 1) && (str2double(initType) ~= 2))
   
    fprintf('\n Select initialization method:\n');
    fprintf('1. PCA\n');
    fprintf('2. SAE\n');
    
    initType = input('', 's');
    
end
initType = str2double(initType);

%% additional variables

% initialization of error estimates
E = zeros(K, len);

time_start = tic;

%% K-fold CV loop

% indices of CV partitions
ind = crossvalind('Kfold', obs, K);


for i = 1 : K

    testing_ind = (ind == i);        % indices for the K fold used for evaluation
    training_ind = ~testing_ind;     % indices for the (K - 1) folds used for training

    train_images_CV = inputs(:, training_ind);
    train_targets_CV = targets(:, training_ind);
    reconstructions_train_targets = reconstructions_targets(:, training_ind);

    test_images_CV = inputs(:, testing_ind);    
    reconstructions_test_targets = reconstructions_targets(:, testing_ind);

    fprintf('\n ------------------->  CV: Iter %d of %d  <------------------- \n', i, K);

    for h = 1 : length(H)

        fprintf('H = %d ...\n', H(h));
        fprintf('\n');

        if (numOfHL == 1)       % 1 hidden layer

            architectureVector(2) = H(h);

        else                    % 2 hidden layers

            architectureVector(3) = H(h);

        end


        if (initType == 1)          % PCA initialization

            initCellVector = PCAInitPrediction(architectureVector(2 : end), 1/40, train_images_CV, 'logsig', 'im');

            net = customizeNetwork(architectureVector, 'logsig', 'im', initCellVector, 500,  'cv');

        else                        % SAE initialization

            net = SAEInit(architectureVector, train_images_CV, 500);

        end


        %% Training

        net = train(net, train_images_CV, train_targets_CV);

        %% Testing on the Kth inner fold

        outputs_test_targets = net(test_images_CV);
        N = size(outputs_test_targets, 2);

        uncompressed_outputs_test_targets = coeff_targets(:, (1 : PCs_M/2))*outputs_test_targets + mu_targets(:, 1 : N);
        E(i, h) = estimateTotalMSE(reconstructions_test_targets, uncompressed_outputs_test_targets, 'im');
        %plotComparison(reconstructions_test_targets, uncompressed_outputs_test_targets, [1 5], architectureVector, reshapeVector, E(i, h), 'cv');

    end

end
time_end = toc(time_start);

% time required for the CV loop
t = datevec(time_end./(60*60*24));
fprintf('\n Time elapsed for %d-fold CV: %.2f (hrs) %.2f (min) %.2f (sec)\n', K, t(4), t(5), t(6));


E_avg = mean(E); 
[r, c] = find(E_avg(1, :) == min(E_avg(1, :)));
E_avg_star = min(E_avg);

if (numOfHL == 1)       % 1 hidden layer
                
    architectureVector(2) = H(c);

else                    % 2 hidden layers

    architectureVector(3) = H(c);

end

%% MSE plot of Cross Validation

figure();
plot(H, E_avg, 'c*-', 'linewidth', 1.5);
hold on;
scatter(H(c), E_avg(c), 'm*', 'linewidth', 1.5);
hold off;
xlim([(min(H) - 10) (max(H) + 10)]);
ylim([(min(E_avg) - min(E_avg)/10^4) (max(E_avg) + max(E_avg)/10^4)]);
xlabel('H', 'fontweight', 'bold');
ylabel('MSE_{avg}', 'fontweight', 'bold');
grid on;
if (numOfHL == 1)
    title(sprintf('Average MSE from %d-fold Cross Validation\n \n {%d-%d-%d}\n\n', K, architectureVector(1), architectureVector(2), architectureVector(3)), 'fontweight', 'bold');
else    
    title(sprintf('Average MSE from %d-fold Cross Validation\n \n {%d-%d-%d-%d}\n\n', K, architectureVector(1), architectureVector(2), architectureVector(3), architectureVector(4)), 'fontweight', 'bold');
end

%% Train Deep Network using optimal H from k-fold CV

if (initType == 1)              % PCA initialization
               
    initCellVector = PCAInitPrediction(architectureVector(2 : end), 1/40, inputs, 'logsig', 'im');

    net = customizeNetwork(architectureVector, 'logsig', 'im', initCellVector, 500);
    
    
else                            % SAE initialization

    net = SAEInit(architectureVector, inputs, 500);

end

net = train(net, inputs, targets);

%% Test trained network on the seen training dataset

outputs_predictions_training = net(inputs);
N = size(outputs_predictions_training, 2);
uncompressed_outputs_predictions_training = coeff_targets(:, (1 : PCs_M/2))*outputs_predictions_training + mu_targets(:, 1 : N);
totalMSE_training = estimateTotalMSE(reconstructions_targets, uncompressed_outputs_predictions_training, 'im');
if (movie == 1)

    plotComparison(reconstructions_targets, uncompressed_outputs_predictions_training, [71 75], architectureVector, reshapeVector, totalMSE_training);
    
else
    
    plotComparison(reconstructions_targets, uncompressed_outputs_predictions_training, [31 35], architectureVector, reshapeVector, totalMSE_training);
    
end
fprintf('\n -------------------- Training Performance --------------------\n');
fprintf('Total MSE on the training dataset: %.5f\n', totalMSE_training);

%% PCA on test data

[coeff_inputs_t, score_inputs_t, latent_inputs_t, mu_inputs_t, totalVarPCs_M_inputs_t, reconstructions_inputs_t, totalMSE_inputs_t] = PCAonData(test_images, PCs_M, mode);

% compressed inputs
inputs_t = score_inputs_t(:, (1 : PCs_M))';


[coeff_predictions_t, score_predictions_t, latent_predictions_t, mu_predictions_t, totalVarPCs_M_predictions_t, reconstructions_predictions_t, totalMSE_predictions_t] = PCAonData(test_predictions, PCs_M/2, mode);

%% Test trained network on the unseen test dataset

outputs_predictions_test = net(inputs_t);
N = size(outputs_predictions_test, 2);
uncompressed_outputs_predictions_test = coeff_predictions_t(:, (1 : PCs_M/2))*outputs_predictions_test + mu_predictions_t(:, 1 : N);
totalMSE_test = estimateTotalMSE(reconstructions_predictions_t, uncompressed_outputs_predictions_test, 'im');

if (movie == 1)
    
    plotComparison(reconstructions_predictions_t, uncompressed_outputs_predictions_test, [61 65], architectureVector, reshapeVector, totalMSE_test);
    plotComparison(reconstructions_predictions_t, uncompressed_outputs_predictions_test, [71 75], architectureVector, reshapeVector, totalMSE_test);
    plotComparison(reconstructions_predictions_t, uncompressed_outputs_predictions_test, [81 85], architectureVector, reshapeVector, totalMSE_test);
    
else
    
    plotComparison(reconstructions_predictions_t, uncompressed_outputs_predictions_test, [21 25], architectureVector, reshapeVector, totalMSE_test);
    plotComparison(reconstructions_predictions_t, uncompressed_outputs_predictions_test, [41 45], architectureVector, reshapeVector, totalMSE_test);
    plotComparison(reconstructions_predictions_t, uncompressed_outputs_predictions_test, [51 55], architectureVector, reshapeVector, totalMSE_test);
    
end
fprintf('\n -------------------- Testing Performance --------------------\n');
fprintf('Total MSE on the test dataset: %.5f\n', totalMSE_test);
