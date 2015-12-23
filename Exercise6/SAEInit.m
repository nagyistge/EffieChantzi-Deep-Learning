          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          %% Chantzi Efthymia - Deep Learning - Exercise 6  %%
          %%                  Tasks A, B                    %%
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function initializes the weights and biases of a deep feedforward%
% neural network with 1, 2 or 3 hidden layers based on ordinary stacked %
% autoencoder(SAE) initialization.                                      %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% architecture: row vector with the network structure, including input, %
% hidden and output layers, as elements                                 %
% activationFunction: activation function of the neurons in hidden      % 
% layers. Two options as an inserted string: 'logsig' or 'tansig'.      %
% inputs: input datasets as a m-by-n matrix, where m(rows) respond to   %
% dimensions and n(columns) respond to the examples/observations        %
% max_epochs: maximum number of training iterations for the newly       %
% created network                                                       %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% net: network object with appropriately initialized weights and biases %
% as well as some other training parameters. The returned network object%
% can be used for fine-tuning                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function net = SAEInit(architecture, inputs, max_epochs)

% length of the row vector with the network structure
len = length(architecture);

if (len == 3)               % 1 hidden layer (3 layers in total)
    
    %% Train 1st Autoencoder
    autoenc1 = trainAutoencoder(inputs, architecture(2), ...     
                                'MaxEpochs', max_epochs, ...
                                'L2WeightRegularization', 0.004, ...
                                'SparsityRegularization', 4, ...
                                'SparsityProportion', 0.15, ...
                                'ScaleData', false);
     
    % extract encoded features from 1st trained autoencoder
    features1 = encode(autoenc1, inputs);
    
    %% Train 2nd Autoencoder
    autoenc2 = trainAutoencoder(features1, architecture(3), ...
                                'MaxEpochs', max_epochs, ...
                                'L2WeightRegularization', 0.004, ...
                                'SparsityRegularization', 4, ...
                                'SparsityProportion', 0.15, ...
                                'ScaleData', false);
                            
    
    %% Deep Neural Network
    net = stack(autoenc1, autoenc2);
    
    
elseif (len == 4)           % 2 hidden layers (4 layers in total)
    
    
    %% Train 1st Autoencoder
    autoenc1 = trainAutoencoder(inputs, architecture(2), ...     
                                'MaxEpochs', max_epochs, ...
                                'L2WeightRegularization', 0.004, ...
                                'SparsityRegularization', 4, ...
                                'SparsityProportion', 0.15, ...
                                'ScaleData', false);
     
    % extract encoded features from 1st trained autoencoder
    features1 = encode(autoenc1, inputs);
    
    %% Train 2nd Autoencoder
    autoenc2 = trainAutoencoder(features1, architecture(3), ...
                                'MaxEpochs', max_epochs, ...
                                'L2WeightRegularization', 0.004, ...
                                'SparsityRegularization', 4, ...
                                'SparsityProportion', 0.15, ...
                                'ScaleData', false);
   
    % extract encoded features from 2nd trained autoencoder
    features2 = encode(autoenc2, features1);
    
    %% Train 3rd Autoencoder
    autoenc3 = trainAutoencoder(features2, architecture(4), ...
                                'MaxEpochs', max_epochs, ...
                                'L2WeightRegularization', 0.004, ...
                                'SparsityRegularization', 4, ...
                                'SparsityProportion', 0.15, ...
                                'ScaleData', false);

    %% Deep Neural Network
    net = stack(autoenc1, autoenc2, autoenc3);
    
elseif (len == 5)        % 3 hidden layers (5 layers in total)
    
    %% Train 1st Autoencoder
    autoenc1 = trainAutoencoder(inputs, architecture(2), ...     
                                'MaxEpochs', max_epochs, ...
                                'L2WeightRegularization', 0.004, ...
                                'SparsityRegularization', 4, ...
                                'SparsityProportion', 0.15, ...
                                'ScaleData', false);
     
    % extract encoded features from 1st trained autoencoder
    features1 = encode(autoenc1, inputs);
    
    %% Train 2nd Autoencoder
    autoenc2 = trainAutoencoder(features1, architecture(3), ...
                                'MaxEpochs', max_epochs, ...
                                'L2WeightRegularization', 0.004, ...
                                'SparsityRegularization', 4, ...
                                'SparsityProportion', 0.15, ...
                                'ScaleData', false);
   
    % extract encoded features from 2nd trained autoencoder
    features2 = encode(autoenc2, features1);
    
    %% Train 3rd Autoencoder
    autoenc3 = trainAutoencoder(features2, architecture(4), ...
                                'MaxEpochs', max_epochs, ...
                                'L2WeightRegularization', 0.004, ...
                                'SparsityRegularization', 4, ...
                                'SparsityProportion', 0.15, ...
                                'ScaleData', false);
    
   % extract encoded features from 3rd trained autoencoder
    features3 = encode(autoenc3, features2);

    %% Train 4th Autoencoder 
    autoenc4 = trainAutoencoder(features3, architecture(5), ...
                                'MaxEpochs', max_epochs, ...
                                'L2WeightRegularization', 0.004, ...
                                'SparsityRegularization', 4, ...
                                'SparsityProportion', 0.15, ...
                                'ScaleData', false);
    
    %% Deep Neural Network
    net = stack(autoenc1, autoenc2, autoenc3, autoenc4);
    
 
else
   
    error('Invalid number of layers\n');
    
end

%% Fine Tuning Common Parameters
net.divideFcn = 'dividerand';
net.trainFcn = 'trainscg';
net.trainParam.epochs = max_epochs;

end
