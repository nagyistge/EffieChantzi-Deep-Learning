        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise 4  %%
        %%                    Task A                      %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script pertains to the implementation of task A, where stacked   %   
% autoencoders are used for weights and biases initialization of a      %
% 3-hidden layer deep  neural network.                                  %
% A figure including the first five original vs. reconstructed images   %
% from the diggitest_dataset is produced with the total mean squared    %
% error of reconstruction on this whole dataset.                        %
% If the neurons in the 2nd hidden layer are three, then a 3-D plot of  %
% their response across the whole test dataset is obtained.             %
%                                                                       %                                                                      
%                                                                       %
% Run this script and a menu will guide you through. More precisely,    % 
% the size of the 3 hidden layers as a row vector is requested.         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
close all;
clear all;

%% Load training dataset into memory

[XTrain, LTrain] = digittrain_dataset;

% transform the cell array of XTrain images into a matrix of vectors
images = createInputs(XTrain, [1 5000]);
dim = size(images, 1);

%% Request user-defined hidden layer sizes

hiddenSize = 0;
while ((sum(mod(hiddenSize, 1)) > 0) || (sum(hiddenSize < 1) > 0) || (length(hiddenSize) ~= 3) || (isempty(hiddenSize)))

    hiddenSize = input('Enter the size of the 3 hidden layers as a row vector: \n');
    
end

%% Train 1st Autoencoder
autoenc1 = trainAutoencoder(images, hiddenSize(1), ...     
                            'MaxEpochs', 1500, ...
                            'L2WeightRegularization', 0.004, ...
                            'SparsityRegularization', 4, ...
                            'SparsityProportion', 0.15, ...
                            'ScaleData', false);

view(autoenc1)

% extract encoded features from 1st trained autoencoder
features1 = encode(autoenc1, images);

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
autoenc4 = trainAutoencoder(features3, dim, ...
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
deepnet = train(deepnet, images, images);

reconstructedTraining = deepnet(images);
totalMSEtraining = estimateTotalMSE(images, reconstructedTraining);

%% Test Network on Test Dataset

[XTest, LTest] = digittest_dataset;
imagesTest = createInputs(XTest, [1 5000]);

reconstructedImagesTest = deepnet(imagesTest);

% total MSE on the whole test dataset
totalMSE = estimateTotalMSE(imagesTest, reconstructedImagesTest);

% visual inspection of first 5 original vs. reconstructed test images
plotComparison(imagesTest, reconstructedImagesTest, [1 5], [dim hiddenSize dim], totalMSE);


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
                          
                    
%% Neurons response manual estimation in the 2nd hidden layer

y2 = 1./(1 + exp(-(W2* (1./(1 + exp(-(W1*images - b1)))) - b2)));
                          
%% 3D Plot of neurons response in the 2nd hidden layer (only with 3 neurons) 

if(hiddenSize(2) == 3)
    
    architectureVector = [dim hiddenSize dim];
    g = sprintf('%d-', architectureVector);
    % plot input vs. reconstructed data 
    figure();
    plot3(y2(1, :), y2(2, :), y2(3, :), 'm*', 'MarkerSize', 10);
    grid on;
    xlabel('neuron_1', 'fontweight', 'bold');
    ylabel('neuron_2', 'fontweight', 'bold');
    zlabel('neuron_3', 'fontweight', 'bold');
    title(sprintf('Neurons response in 2^{nd} hidden layer of Deep network %s\n After auto-encoders initialization', g(1 : (end - 1))));

end
