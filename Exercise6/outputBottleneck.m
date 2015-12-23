          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          %% Chantzi Efthymia - Deep Learning - Exercise 6 %%
          %%                     Task  B                   %%
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function estimates the outputs from the bottleneck node layer    %
% of the deep neural network, which was successfully trained on the     %
% time-lapse microscopy movies during Exercise 5.                       %
% This function is nested in the function 'generateImageData_taskB',for %
% the calculation of the mean vector m_t and covariance matrix C_t.     %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% WeightsBiases: cell array containing the weights and biases of the    %
% first and second layer of the deep network trained on the time-lapse  %
% microscopy movies during Exercise 5                 %
% data: time-lapse microscopy frame dataset used for the training of    %
% the deepnet network                                                   %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% output: n-dimensional feature vectors t_n, which come out from the    %
% bottleneck node layer of the deep network trained during Exercise 5   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function output = outputBottleneck(WeightsBiases, data)

obs = size(data, 2);

b1 = repmat(WeightsBiases{1, 5}, 1, obs);
b2 = repmat(WeightsBiases{1, 6}, 1, obs);

output = 1./(1 + exp(-(WeightsBiases{1, 2}* (1./(1 + exp(-(WeightsBiases{1, 1}*data - b1)))) - b2)));

end