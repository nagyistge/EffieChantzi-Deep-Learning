
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise 2  %%
        %%                    Tasks A, B                  %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script executes the function "testClassifierOnce" user-defined   %
% times on digittrain_dataset and digittest_dataset. The reason for     %
% this is that during the training of the pattern recognition network,  %
% the training dataset is randomly divided into training(70%),          %
% validation(15%) and test(15%) tests. Hence, it is more reliable to    %
% perform the training procedure a couple of times, evaluate its        %
% performance at each iteration and finally, calcutate the mean false   %
% alarm and detection probabilities.                                    %
%                                                                       %
%                                                                       %
%                                                                       %
% Run this script and enter the size of hidden layer(s), the number of  %
% iterations to evaluate the classification performance, as well as     %
% if you want to change from the default performance measure            %
% 'crossentropy' to 'mse'.                                              %
%                                                                       %
%                                                                       %
% IMPORTANT!!                                                           %
% In case of multiple hidden layers, when asked, enter them as a row    %
% vector. Example: [25 10] -> indicates two hidden layers, 25 and 10    %
% neurons in the first and second hidden layer, respectively.           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc;
close all;
clear all;

%load training dataset
[XTrain, LTrain] = digittrain_dataset;

% load test dataset
[XTest, LTest] = digittest_dataset;

% all images, from 1 to 5000
N = [1 5000];

fprintf('--------------------------- Classification Performance --------------------------\n');
hiddenSize = 0;
while ((sum(mod(hiddenSize, 1)) > 0) || (sum(hiddenSize < 1) > 0) || (isempty(hiddenSize)))

    hiddenSize = input('Enter the size of hidden layers: \n');
    
end

iter = 0;
while ((sum(mod(iter, 1)) > 0) || (sum(iter < 1) > 0) || (isempty(iter)))

    iter = input('Enter the number of iterations for performance evaluation: \n');
    
end

performFcn = '';
while ((~strcmpi(performFcn, 'Y') && ~strcmpi(performFcn, 'N')) || (isempty(performFcn)))
    
    performFcn = input('Change from default "crossentropy" to "mse"? (Y/N):\n', 's');
    
end


probabilities = zeros(2, iter);

for i = 1 : iter
    
    fprintf('--------------------------------- Iteration %d ---------------------------------\n', i);
    if (strcmpi(performFcn, 'Y'))
        probs = testClassifierOnce(XTrain, LTrain, XTest, LTest, N, hiddenSize, 'mse');
    else
        probs = testClassifierOnce(XTrain, LTrain, XTest, LTest, N, hiddenSize);
    end
    
    probabilities(1, i) = probs(1);
    probabilities(2, i) = probs(2);
    fprintf('--------------------------------------------------------------------------------\n\n');
    
end

fprintf('\n');
fprintf('Mean False Alarm Probability: %.2f%%\n', mean(probabilities(1, :)));
fprintf('Mean Detection Probability: %.2f%%\n', mean(probabilities(2, :)));
