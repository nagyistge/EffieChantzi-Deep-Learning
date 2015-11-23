        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise 3  %%
        %%                    Tasks F                     %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script pertains to the implementation of task F, where PCA is    %
% used for weights and biases initialization of a 3-hidden layer deep   %
% neural network. In this case, the training set consists only of 200   %
% samples; 100 images showing "3" and 100 images showing "other".       %
% However, the testing procedure on the digittest_dataset can be        %
% performed either by the default activated or user-defined deactivated %
% built-in "early stopping" procedure.                                  %
%                                                                       %
%                                                                       %
% Run this script and a menu will guide you through. More precisely,    % 
% the size of the 3 hidden layers as a row vector, the type of the      %
% neurons' activation function and the deactivation of the by default   %
% activated "early stopping" are requested. Valid options for activation%
% function: either 'logsig' or 'tansig'.                                %
%                                                                       %
%                                                                       %
% IMPORTANT !!!                                                         %
% Here, s = 1/40 (line 90).                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
close all;
clear all;

%% Load trainning dataset into memory

[XTrain, LTrain] = digittrain_dataset;

% transform the cell array of XTrain images into a matrix of vectors taking
% only the 5 first images
images = createInputs(XTrain, [1 5000]);
dim = size(images, 1);

% transform label data of train images into the proper format
labels = createNewLabels(LTrain, [1 5000]);

% indexes of test images with digit "3" and "others"
[r_1, c_1] = find(labels == 1);
[r_0, c_0] = find(labels == 0);

%% New Training dataset consisting only of 200 samples: 100 "3" & 100 "others"
images_3 = zeros(dim, 100);
for i = 1 : 100
    images_3(:, i) = images(:, c_1(i));
end

images_others = zeros(dim, 100);
for i = 1 : 100
    images_others(:, i) = images(:, c_0(i));
end

images_200 = zeros(dim, 200);
images_200(:, 1 : 100) = images_3;
images_200(:, 101 : 200) = images_others;

%% Request user-defined hidden layer sizes

hiddenSize = 0;
while ((sum(mod(hiddenSize, 1)) > 0) || (sum(hiddenSize < 1) > 0) || (isempty(hiddenSize)))

    hiddenSize = input('Enter the size of the 3 hidden layers as a row vector: \n');
    
end

%% Request user-defined activation function
activationFunction = '';
while ((~strcmpi(activationFunction, 'logsig') && ~strcmpi(activationFunction, 'tansig')) || (isempty(activationFunction)))
    
    activationFunction = input('Enter the name of the transfer function to be used (logsig/tansig):\n', 's');
    
end

%% Request deactivation of built-in early stopping

earlyStopping = '';
while ((~strcmpi(earlyStopping, 'Y') && ~strcmpi(earlyStopping, 'N')) || (isempty(earlyStopping)))
    
    earlyStopping = input('Deactivate early stopping? (Y/N):\n', 's');
    
end

%% PCA Initialization

[W1, W2, W3, W4, b1, b2, b3, b4] = PCAInitialization([hiddenSize(1) hiddenSize(2)], 1/40, images, activationFunction);

%% Weights and biases initialization 

architectureVector = [dim hiddenSize dim];
net = customizeNetwork(architectureVector, W1, W2, W3, W4, b1, b2, b3, b4, activationFunction);

%% Training

net.trainParam.epochs = 1500;

% turn off Matlab's early stopping if user requests so
if (strcmpi(earlyStopping, 'Y'))

    net.trainParam.max_fail = 100; 
    
end

net = train(net, images_200, images_200);

%% Load Test dataset into memory

[XTest, LTest] = digittest_dataset;
imagesTest = createInputs(XTest, [1 5000]);

%% Test trained network on test dataset
outputs = net(imagesTest);

% total reconstruction error
totalMSE = estimateTotalMSE(imagesTest, outputs);

% visual inspection of 5 original vs. reconstructed test  images
plotComparison(imagesTest, outputs, [1 5], architectureVector, totalMSE);

