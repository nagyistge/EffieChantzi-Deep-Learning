          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          %% Chantzi Efthymia - Deep Learning - Exercise 6 %%
          %%                     Task  B                   %%
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates the d-dimensional F_image generated data from   %
% the compressed n-dimensional(n < d) feature vectors t_n, which are    %
% drawn from a normal distribution with mean m_t and variance based on  %
% the covariance matrix C_t, coming out of the bottleneck node layer    %
% of the trained network on the time-lapse microscopy movies from       %
% Exercise 5.                                                           %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% deepnet: network object of the deep network trained successfully on   %
% on the time-lapse microscopy movies during Exercise 5                 %
% data: time-lapse microscopy frame dataset used for the training of    %
% the deepnet network                                                   %
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
% to the feedforward neural network F_image                             %
% outputs: d-dimensional output vectors x_n from the F_image network    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [dataset, outputs] = generateImageData_taskB(deepnet, data, architecture, activationFunction, obs, threshold)

% initialization of cell array for weights and biases
WeightsBiases = cell(1, 2*deepnet.numLayers);

% extract weights and biases from the fine-tuned deepnet network (Exercise 5)
WeightsBiases{1, 1} = deepnet.IW{1, 1};
WeightsBiases{1, 2} = deepnet.LW{2, 1};
WeightsBiases{1, 3} = deepnet.LW{3, 2};
WeightsBiases{1, 4} = deepnet.LW{4, 3};
WeightsBiases{1, 5} = deepnet.b{1, 1};
WeightsBiases{1, 6} = deepnet.b{2, 1};
WeightsBiases{1, 7} = deepnet.b{3, 1};
WeightsBiases{1, 8} = deepnet.b{4, 1};

% initialization of cell array for weights and biases of the second half of
% the deep neural network from Exercise 5
initCellVector = cell(1, 2*(length(architecture) - 1));
initCellVector{1, 1} = WeightsBiases{1, 3};
initCellVector{1, 2} = WeightsBiases{1, 4};
initCellVector{1, 3} = WeightsBiases{1, 7};
initCellVector{1, 4} = WeightsBiases{1, 8};

% customize feedforward network for the data generation
net = customizeNetwork(architecture, activationFunction, 'im', initCellVector, 0);

% extraction of the outputs from the 3rd layer (bottleneck node)
y3 = outputBottleneck(WeightsBiases, data);

% mean vector
m_vector = mean(y3, 2); 

% standard deviation vector
C_vector = std(y3, 0, 2);

dataset = zeros(architecture(1), obs);

% normal distribution
randomMatrix = randn(architecture(1), obs);

for i = 1 : obs

    % random vectors drawn from a normal distribution with mean m_t and
    % standard deviation based on the covariance matrix C_t
    dataset(:, i) = m_vector + C_vector.*randomMatrix(:, i);
    
end

% extract weights and biases for checking against saturation events
W1 = initCellVector{1, 1};
b1 = initCellVector{1, 3};
b1 = repmat(b1, 1, obs);

W2 = initCellVector{1, 2};
b2 = initCellVector{1, 4};
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
    fprintf('Minimum standard deviation of hidden layer: %.8f\n', min(std1));
    fprintf('Minimum standard deviation of output layer: %.8f\n', min(std2));
    fprintf('\n');
    
else
    
    error('Saturation');
    
end

end
