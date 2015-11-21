        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise 3  %%
        %%                    Task B                      %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script pertains to the implementation of task B, where a         %
% "bottle-neck" neural network of stacked autoencoders with 3 user      % 
% defined hidden layers is employed and trained. The goal is to compress%
% the input images down to fewer dimensions than 'M'(also user-defined),%
% as obtained with PCA, while retaining the same total mean             % 
% reconstruction error.                                                 %
% A figure with the first 5 original vs. reconstructed test images, as  %
% well as the total mean square error is obtained.                      %
% Finally, if there are 3 neurons in the 2nd hidden layer, a 3-D plot   %
% showing their response on 100 images showing "3" and 100 images       %
% showing "other" from the whole test dataset is displayed.             %
%                                                                       %
%                                                                       %
% Run this script and a menu will guide you through. More precisely,    % 
% the number of the 'M' principle components used for the initial       %
% compression as well as as the size of the three hidden layers are     %
% are requested from the user. The size of the hidden layers must be    %
% entered as a row vector (e.g., [15 3 15]).                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc;
close all;
clear all;

%% Load training dataset into memory

[XTrain, LTrain] = digittrain_dataset;

% transform the cell array of XTrain images into a matrix of vectors taking
% only the 5 first images
images = createInputs(XTrain, [1 5000]);

%% Request user-defined reduced dimensions

% principal components of the M reduced dimensions
PCs_M = 0;
while ((PCs_M <= 0) || (mod(PCs_M, 1) ~= 0))

    PCs_M = input('Enter a positive value for the first M Principal Components(reduced dimensions): \n');
    
end

%% PCA on training data

[~, score, ~, ~, ~, ~, ~] = PCAonImages(images, PCs_M);

t_n = score(:, (1 : PCs_M))';


hiddenSize = 0;
while ((sum(mod(hiddenSize, 1)) > 0) || (sum(hiddenSize < 1) > 0) || (isempty(hiddenSize)))

    hiddenSize = input('Enter the size of hidden layers as a row vector: \n');
    
end


%% Train 1st Autoencoder
autoenc1 = trainAutoencoder(t_n, hiddenSize(1), ...     
                            'MaxEpochs', 1500, ...
                            'L2WeightRegularization', 0.004, ...
                            'SparsityRegularization', 4, ...
                            'SparsityProportion', 0.15, ...
                            'ScaleData', false);

view(autoenc1)

% extract encoded features from 1st trained autoencoder on the compressed set
features1 = encode(autoenc1, t_n);

%% Train 2nd Autoencoder
autoenc2 = trainAutoencoder(features1, hiddenSize(2), ...
                            'MaxEpochs', 1500, ...
                            'L2WeightRegularization', 0.004, ...
                            'SparsityRegularization', 4, ...
                            'SparsityProportion', 0.15, ...
                            'ScaleData', false);

view(autoenc2)

% extract encoded features from 2nd trained autoencoder
features2 = encode(autoenc2, features1);

%% Train 3rd Autoencoder
autoenc3 = trainAutoencoder(features2, hiddenSize(3), ...
                            'MaxEpochs', 1500, ...
                            'L2WeightRegularization', 0.004, ...
                            'SparsityRegularization', 4, ...
                            'SparsityProportion', 0.15, ...
                            'ScaleData', false);

view(autoenc3)

% extract encoded features from 3rd trained autoencoder
features3 = encode(autoenc3, features2);

%% Train 4th Autoencoder 
autoenc4 = trainAutoencoder(features3, size(t_n, 1), ...
                            'MaxEpochs', 1500, ...
                            'L2WeightRegularization', 0.004, ...
                            'SparsityRegularization', 4, ...
                            'SparsityProportion', 0.15, ...
                            'ScaleData', false);

view(autoenc4)

% extract encoded features from 4th trained autoencoder
features4 = encode(autoenc4, features3);

%% Deep Neural Network

deepnet = stack(autoenc1, autoenc2, autoenc3, autoenc4);
view(deepnet)

%% Fine Tuning/Supervised Learning

deepnet.trainParam.epochs = 1500;
deepnet = train(deepnet, t_n, t_n);


%% Test Data

[XTest, LTest] = digittest_dataset;
imagesTest = createInputs(XTest, [1 5000]);

%% PCA on test data

[coeff, score, latent, mu, totalVarPCs_M, PCAReconstruction, totalMSE] = PCAonImages(imagesTest, PCs_M);
t_nTest = score(:, (1 : PCs_M))';


reconstructed_t_nTest = deepnet(t_nTest);
PCAReconstruction_AE = coeff(:, (1 : PCs_M))*reconstructed_t_nTest + mu;

totalMSE_PCA = totalMSE;
totalMSE_PCA_AE = estimateTotalMSE(imagesTest, PCAReconstruction_AE);

architectureVector = [PCs_M hiddenSize PCs_M];
plotComparison(imagesTest, PCAReconstruction_AE, [1 5], architectureVector, totalMSE_PCA_AE);
plotComparison(imagesTest, PCAReconstruction, [1 5], [], totalMSE_PCA);

%% Extract outputs in Hidden Layers

% extract weights and biases of the supervised trained deep neural network

W1 = deepnet.IW{1}; % weights of 1st hidden layer

W2 = deepnet.LW{2, 1}; % weights of 2nd hidden layer

b1 = deepnet.b{1}; % biases of 1st hidden layer

b1 = repmat(b1, 1, 5000); % biases of 1st hidden layer tranformed into appropriate
                          % matrix for computation of f(W1*X - b1)

b2 = deepnet.b{2}; % biases of 2nd hidden layer

b2 = repmat(b2, 1, 5000); % biases of 2nd hidden layer tranformed into appropriate
                          % matrix for computation of f(W2*f(W1*X - b1)) -b2)
                          
       
%% Neurons response in the second hidden layer

y = 1./(1 + exp(-(W2* (1./(1 + exp(-(W1*t_nTest - b1)))) - b2)));   


%% 3D-plot of outputs of 2nd hidden layer if there are 3 neurons: 100 samples of "3" and 100 of "others"

if (hiddenSize(2) == 3)

    % transform label data into the proper format
    labels = createNewLabels(LTest, [1 5000]);


    % indexes of test images with digit "3" and "others"
    [r_1, c_1] = find(labels == 1);

    [r_0, c_0] = find(labels == 0);


    samples_3 = zeros(3, 100);
    for i = 1 : 100

        samples_3(: , i) = y(:, c_1(i));

    end

    samples_others = zeros(3, 100);
    for i = 1 : 100

        samples_others(: , i) = y(:, c_0(i));

    end

    % plot neurons' 3-D response
    g = sprintf('%d-', architectureVector);
    figure();
    plot3(samples_3(1, :), samples_3(2, :), samples_3(3, :), 'm*', 'MarkerSize', 10);
    hold on;
    plot3(samples_others(1, :), samples_others(2, :), samples_others(3, :), 'go', 'MarkerSize', 10);
    grid on;
    xlabel('neuron_1', 'fontweight', 'bold');
    ylabel('neuron_2', 'fontweight', 'bold');
    zlabel('neuron_3', 'fontweight', 'bold');
    title(sprintf('3-D outputs from 2^{nd} hidden layer of Deep network %s', g(1 : (end - 1))));
    legend({'3', 'others'}, 'Location', 'northeast', 'FontSize', 8, 'FontWeight', 'bold');
    hold off;    

end