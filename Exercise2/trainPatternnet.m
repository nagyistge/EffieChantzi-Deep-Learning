        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise 2  %%
        %%                    Tasks A, B                  %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates and trains a pattern recognition network, which % 
% is a category of feedforward networks, to classify inputs according   %
% to target classes. It uses the built-in matlab function "patternnet". %
% Moreover, it tests the performance of the trained network on the test %
% set(15%) of the training set(70%) and prints the false alarm and      %
% detection probability.                                                %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% hiddenSize: number of neurons in the hidden layer                     %
% inputs: input data in the form of matrix, where rows correspond to    % 
% features and columns to examples(i.e., images)                        %
% targets:  target/label data as vectors with "1" in the class they     %
% represent                                                             %
% varargin: optional user-defined argument for setting the performance  % 
% function to 'mse'                                                     %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% net:  trained pattern recognition network                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function net = trainPatternnet(hiddenSize, inputs, targets, varargin)

if (nargin < 3)
    
    error('Invalid number of inputs');
        
else
    
    net = patternnet(hiddenSize);
    if ((length(varargin) == 1) && (strcmpi(varargin{1}, 'mse') == 1))
            
            net.performFcn = 'mse';
            
    end
                
end
    
% train NN 
net = train(net, inputs, targets);

% test network on the test set of training data
outputs = net(inputs);

[~, ~, ~, probabilities] = confusion(targets, outputs);

fprintf('----------> Test Set of Training Dataset <----------\n');
fprintf('False alarm probability: %.2f%%', probabilities(1, 2)*100);
fprintf('\n');
fprintf('Detection probability: %.2f%%', probabilities(2, 3)*100);
fprintf('\n');
    
end
