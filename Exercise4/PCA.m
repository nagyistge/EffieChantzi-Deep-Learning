        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise 4  %%
        %%                    Task A                      %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script pertains to the implementation of task A, where PCA is    %
% used for weights and biases initialization of a 3-hidden layer deep   %
% neural network for image reconstruction.                              %
% A figure including the first five original vs. reconstructed images   %
% from the diggitest_dataset is produced with the total mean squared    %
% error of reconstruction on this whole dataset.                        %
% If the neurons in the 2nd hidden layer are three, then a 3-D plot     %
% of their response across the whole test dataset is obtained.          %
%                                                                       %
%                                                                       %
%                                                                       %
% Run this script and a menu will guide you through. More precisely,    % 
% the size of the 3 hidden layers as a row vector, as well as the type  %
% of the neurons' activation function are requested. Valid options for  %
% activation function: either 'logsig' or 'tansig'.                     %
%                                                                       %
%                                                                       %
% IMPORTANT !!!                                                         %
% Here, s = 1/40 (line 55).                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
close all;
clear all;

%% Load trainning dataset into memory

[XTrain, LTrain] = digittrain_dataset;

% transform the cell array of XTrain images into a matrix of vectors
images = createInputs(XTrain, [1 5000]);
dim = size(images, 1);

%% Request user-defined hidden layer sizes

hiddenSize = 0;
while ((sum(mod(hiddenSize, 1)) > 0) || (sum(hiddenSize < 1) > 0) || (length(hiddenSize) ~= 3) || (isempty(hiddenSize)))

    hiddenSize = input('Enter the size of the 3 hidden layers as a row vector: \n');
    
end

activationFunction = '';
while ((~strcmpi(activationFunction, 'logsig') && ~strcmpi(activationFunction, 'tansig')) || (isempty(activationFunction)))
    
    activationFunction = input('Enter the name of the transfer function to be used (logsig/tansig):\n', 's');
    
end

%% PCA Initialization

[W1, W2, W3, W4, b1, b2, b3, b4] = PCAInitialization([hiddenSize(1) hiddenSize(2)], 1/40, images, activationFunction);

%% Weights and biases initialization 

architectureVector = [dim hiddenSize dim];
net = customizeNetwork(architectureVector, W1, W2, W3, W4, b1, b2, b3, b4, activationFunction);

%% Training

net.trainParam.epochs = 1500;
net = train(net, images, images);

%% Load Test dataset into memory

[XTest, LTest] = digittest_dataset;
imagesTest = createInputs(XTest, [1 5000]);

%% Test trained network on test dataset
outputs = net(imagesTest);

% total reconstruction error
fprintf('\n');
fprintf('-------------------- Testing Performance --------------------\n');
totalMSE = estimateTotalMSE(imagesTest, outputs);
fprintf('Total MSE on the test dataset: %.5f', totalMSE);
fprintf('\n');

% visual inspection of 5 original vs. reconstructed test  images
plotComparison(imagesTest, outputs, [1 5], architectureVector, totalMSE);

%% Manual estimation of neurons response in the 2nd hidden layer

y2 = 1./(1 + exp(-(W2* (1./(1 + exp(-(W1*images - b1)))) - b2)));

%% 3D Plot neurons response in the 2nd hidden layer (only with 3 neurons)

if (hiddenSize(2) == 3)
    
    g = sprintf('%d-', architectureVector);
    % plot input vs. reconstructed data 
    figure();
    plot3(y2(1, :), y2(2, :), y2(3, :), 'm*', 'MarkerSize', 10);
    grid on;
    xlabel('neuron_1', 'fontweight', 'bold');
    ylabel('neuron_2', 'fontweight', 'bold');
    zlabel('neuron_3', 'fontweight', 'bold');
    title(sprintf('Neurons response in 2^{nd} hidden layer of Deep network %s\n After PCA initialization', g(1 : (end - 1))));
    hold off;
    
end

