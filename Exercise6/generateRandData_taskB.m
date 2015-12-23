          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          %% Chantzi Efthymia - Deep Learning - Exercise 6 %%
          %%                     Task  B                   %%
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates the d-dimensional F_rand generated data from the%
% compressed n-dimensional(n < d) feature vectors t_n, which are drawn  %
% from a uniform distribution with zero mean and unity variance.        %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% alpha: user-defined parameter declaring the range of the interval     %
% architecture: row vector with the network structure as elements       % 
% activationFunction: neurons' activation function; either 'logsig' or  %
% 'tansig'                                                              %
% obs: size of generated data; number of examples/observations          %
% threshold: threshold value for the minimum standard deviation of      %
% neurons' outputs in the hidden(s) and output layers. It is used       %
% as a guard against saturation events. If the resulting standard       %
% deviations are equal to or greater than this threshold, the data      %
% generation is aborted.                                                %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% dataset: n-dimensional feature vectors t_n, which are used as inputs  %
% to the feedforward neural network F_rand                              %
% outputs: d-dimensional output vectors x_n from the F_rand network     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [dataset, outputs] = generateRandData_taskB(alpha, architecture, activationFunction, obs, threshold)

% random selected weights and biases from the interval [-a, a]
WeightsBiases = randomIntervalInit(alpha, architecture);

% customize a random-based feedforward network (F_rand) used for data
% generation
net = customizeNetwork(architecture, activationFunction, 'im', WeightsBiases, 0);

% n-dimensional feature vectors t_n drawn from a uniform distribution with
% zero mean and unit variance
dataset = rand(architecture(1), obs);

% weights and biases extraction from the newly created F_rand network, in
% order to check against saturation events
W1 = WeightsBiases{1, 1};
b1 = WeightsBiases{1, 3};
b1 = repmat(b1, 1, obs);

W2 = WeightsBiases{1, 2};
b2 = WeightsBiases{1, 4};
b2 = repmat(b2, 1, obs);

% neurons' response in the hidden layer
y1 = 1./(1 + exp(-(W1*dataset - b1)));

% standard deviation of neurons' response in the hidden layer
std1 = std(y1, 0, 2);

% neurons' response in the output layer
y2 = 1./(1 + exp(-(W2* (1./(1 + exp(-(W1*dataset - b1)))) - b2)));

% standard deviation of neurons' response in the output layer
std2 = std(y2, 0, 2);

% only if the minimum standard deviation of the hidden and output layers is
% above the user-specified threshold, the selected random weights do not
% result in saturations, and thus, the generation of the dataset is
% performed
if ((min(std1) >= threshold) && (min(std2) >= threshold))
   
    outputs = net(dataset);
    
    fprintf('\n');
    fprintf('Minimum standard deviation of hidden layer: %.5f\n', min(std1));
    fprintf('Minimum standard deviation of output layer: %.5f\n', min(std2));
    fprintf('\n');
    
else
    
    error('Saturation due to random weights\n');
    
end


end
