        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercises 3,4  %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates, customizes and initializes a feedforward neural%
% network with 3 hidden layers. The initialization of weights and biases%
% is implemented by performing PCA, as described in task C. Thus,       %
% this function can be used after the execution of the function         %
% "PCAInitialization", which returns the necessary weights and biases   %
% matrices to be used.                                                  %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% architecture: row vector with the dimensions of all layers, including %
% input, hidden and output layers, as elements                          %
% W1: weights matrix for input layer (size_HL1xinitial_dimensions)      %
% W2: weights matrix for 1st hidden layer (size_HL2xsize_HL1)           %
% W3: weights matrix for 2nd hidden layer (size_HL3xsize_HL2)           %
% W4: weights matrix for output layer (initial_dimensionsxsize_HL3)     %
% b1: biases matrix for input layer (size_HL1xobservations)             %
% b2: biases matrix for 1st hidden layer (size_HL2xobservations)        %
% b3: biases matrix for 2nd layer (size_HL3xobservations)               %
% b4: biases matrix for output layer (initial_dimensionsxobservations)  %
% activationFunction: activation function of the neurons in hidden      %
% layers. Two options as an inserted string: 'logsig' or 'tansig'.      %
% varargin: optional user-defined argument for cross validation. This   %
% optional parameter updates the version of this function from Exercise %
% 3. If this parameter is set to 'cv', then the default division        %
% function 'dividerand' is replaced by 'dividetrain', since during      %
% cross-validation the division of data is invoked by the current       %
% user's implementation.                                                %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% net: network object with initialized weights and biases from PCA      %
% initialization as well as some other training parameters. The         %
% returned network object can be used for training and testing.         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function net = customizeNetwork(architecture, W1, W2, W3, W4, b1, b2, b3, b4, activationFunction, varargin)


if (nargin < 10)
    
    error('Invalid number of inputs');
        
elseif (length(varargin) > 1)
    
    error('Invalid number of optional input arguments');
    
elseif ((length(varargin) == 1) && (strcmpi(varargin{1}, 'cv') ~= 1))
    
    error('Invalid type of optional input arguments');
        
else
    
    if (strcmpi(activationFunction, 'logsig') == 1)

        mode = 1;

    elseif (strcmpi(activationFunction, 'tansig') == 1)

        mode = 2;

    else

        error('Invalid type of activation function. Choose between "logsig" and "tansig".\n');

    end

    len = length(architecture);

    net = network(1, (len - 1));

    % set to 1 all layers that have biases
    net.biasConnect = ones((len - 1), 1);

    % only first layer have weights that come from inputs
    net.inputConnect(1, 1) = 1;

    net.layerConnect(2, 1) = 1;
    net.layerConnect(3, 2) = 1;
    net.layerConnect(4, 3) = 1;

    % layer that generate outputs
    net.outputConnect(1, 4) = 1;

    %% customize layers

    % layer 1
    net.layers{1}.dimensions = size(W1, 1);
    net.layers{1}.name = 'Hidden 1';

    % layer 2
    net.layers{2}.dimensions = size(W2, 1);
    net.layers{2}.name = 'Hidden 2';


    % layer 3
    net.layers{3}.dimensions = size(W3, 1);
    net.layers{3}.name = 'Hidden 3';

    % user-defined transfer functions
    if (mode == 1)

        net.layers{1}.transferFcn = 'logsig';
        net.layers{2}.transferFcn = 'logsig';
        net.layers{3}.transferFcn = 'logsig';
        net.layers{4}.transferFcn = 'logsig';

    else

        net.layers{1}.transferFcn = 'tansig';
        net.layers{2}.transferFcn = 'tansig';
        net.layers{3}.transferFcn = 'tansig';
        net.layers{4}.transferFcn = 'tansig';

    end

    % output layer
    net.layers{4}.dimensions = size(W4, 1);
    net.layers{4}.name = 'Output';

    % dimensions of input vectors
    net.inputs{1}.size = architecture(1);

    %% weights initialization

    % 1st layer
    net.IW{1, 1} = W1;

    % 2nd hidden layer
    net.LW{2, 1} = W2;

    % 3rd hidden layer
    net.LW{3, 2} = W3;

    % output hidden layer
    net.LW{4, 3} = W4;

    %% biases initialization

    % 1st layer
    net.b{1, 1} = b1(:, 1);

    % 2nd layer
    net.b{2, 1} = b2(:, 1);

    % 3rd layer
    net.b{3, 1} = b3(:, 1);

    % output layer
    net.b{4, 1} = b4(:, 1);

    %% other parameters

    if (length(varargin) == 1)

        % use all training data from K-1 folds in CV inner loop
        net.divideFcn = 'dividetrain';

    else

        % randomly division to training, validation and test sets of the training
        % dataset
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = 0.8;
        net.divideParam.valRatio = 0.1;
        net.divideParam.testRatio = 0.1;

    end

    % training function
    net.trainFcn = 'trainscg';

end

end