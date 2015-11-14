        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise 2  %%
        %%                 Tasks C, D, E                  %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script trains and tests a deep neural network, consisting of two %
% stacked autoencoders and one softmax layer. The hidden size of the two%
% autoencoders is user-defined. The performance of the deep network is  %
% firstly evaluated on the unsupervised trained network. However, the   %
% deep network is then trained in an supervised way by performing       %
% backpropagation (fine tuning). In this way, the performance can be    %
% improved.                                                             %
% Hence, the perfomance results; false alarm and detection probabilities%
% after fine tuning are also displayed and plotted in the respective    %
% confusion matrix.                                                     %
% The last part of this script is the extraction of the neurons'        %
% response in each hidden layer. In the 1st hidden layer, the weights   %
% are plotted in a common graphical window as images. In the 2nd hidden %
% layer, a (size_of_2ndHiddenLayer)-by-(number_of_images) matrix is     %
% obtained. Each row indicates the response of the respective neuron    % 
% from the input images.                                                % 
%                                                                       %
%                                                                       %
% IMPORTANT!!                                                           %
% The training and test datasets are "digittrain_dataset" and           %
% "digittest_dataset" respectively.                                     %
% Until now, the default 'logsig' is used.                              %
% I will keep this script up-to-dated, since the results so far, with   % 
% [784-25-10-1] indicate that all the neurons in the 2nd hidden layer   %
% but one, have the same very high response(very close to 1) for all    %
% input images. You can view the 2nd layer neurons' response by         %
% opening the variable 'y' from the workspace, after executing this     %
% script.                                                               %
%                                                                       %
%                                                                       %
% Run this script and enter the size of hidden layers of stacked        %
% autoencoders as a row vector (i.e, [25 10]).                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc;
close all;
clear all;

%% Load training dataset into memory

[XTrain, LTrain] = digittrain_dataset;

% all images, from 1 to 5000
N = [1 5000];

fprintf('--------------------------- Deep Neural Network --------------------------\n');
fprintf('Pre-training....\n\n');

hiddenSize = 0;
while ((sum(mod(hiddenSize, 1)) > 0) || (sum(hiddenSize < 1) > 0))

    hiddenSize = input('Enter the size of the 2 hidden layers: \n');
    
end

% transform label data into the proper format
labels = createNewLabels(LTrain, N);

%% Train 1st Autoencoder
autoenc1 = trainAutoencoder(XTrain, hiddenSize(1), ...
                            'MaxEpochs', 1400, ...
                            'L2WeightRegularization', 0.004, ...
                            'SparsityRegularization', 4, ...
                            'SparsityProportion', 0.15, ...
                            'ScaleData', false);

view(autoenc1)

% extract encoded features from 1st trained autoencoder
features1 = encode(autoenc1, XTrain);

%% Train 2nd Autoencoder
autoenc2 = trainAutoencoder(features1, hiddenSize(2), ...
                            'MaxEpochs', 1400, ...
                            'L2WeightRegularization', 0.002, ...
                            'SparsityRegularization', 4, ...
                            'SparsityProportion', 0.15, ...
                            'ScaleData', false);

view(autoenc2)

% extract encoded features from 2nd trained autoencoder
features2 = encode(autoenc2, features1);


%% Train the final softmax layer
 
% supervised training using labels for training data
softnet = trainSoftmaxLayer(features2, labels, 'MaxEpochs', 1400, 'LossFunction', 'crossentropy');
view(softnet)
 
%% Deep Neural Network

% stack the 2 autoencoders and the final softmax layer
deepnet = stack(autoenc1, autoenc2, softnet);
view(deepnet)
 
%% Test deepnet on test data

% load test dataset into memory
[XTest, LTest] = digittest_dataset;

% transform test data into proper format
testImages = createInputs(XTest, N);
testLabels = createNewLabels(LTest, N);


outputs = deepnet(testImages);

[~, ~, ~, probs] = confusion(testLabels, outputs);
figure();
plotconfusion(testLabels, outputs);


fprintf('------> Unsupervised Learning <------\n');
fprintf('False alarm probability: %.2f%%', probs(1, 2)*100);
fprintf('\n');
fprintf('Detection probability: %.2f%%', probs(2, 3)*100);
fprintf('\n\n\n');

%% Fine tuning the deep neural network

fprintf('Global Training....\n\n');

% transform test data into proper format
images = createInputs(XTrain, N);

% perform fine tuning - supervised learning
deepnet = train(deepnet, images, labels);

outputs_fineTuning = deepnet(testImages); 

[~, ~, ~, probs] = confusion(testLabels, outputs_fineTuning);
figure();
plotconfusion(testLabels, outputs_fineTuning);

fprintf('------> Fine Tuning <------\n');
fprintf('False alarm probability: %.2f%%', probs(1, 2)*100);
fprintf('\n');
fprintf('Detection probability: %.2f%%', probs(2, 3)*100);
fprintf('\n');

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

% 1st Hidden Layer
plotWeightsAsImages(W1);

% response of neurons in the 2nd Hidden Layer
% the objective function for each neuron i can be obtained as y(i, :)
y = 1./(1 + exp(-(W2* (1./(1 + exp(-(W1*images - b1)))) - b2)));

%% To be continued.....


