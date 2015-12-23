          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          %% Chantzi Efthymia - Deep Learning - Exercise 6  %%
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates, customizes and initializes feedforward neural  %
% networks with 1, 2 or 3 hidden layers. The initialization of weights  %
% and biases can be either PCA-based or random-based.                   %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% architecture: row vector with the network structure, including input, %
% hidden and output layers, as elements                                 %
% activationFunction: activation function of the neurons in hidden      % 
% layers. Two options as an inserted string: 'logsig' or 'tansig'.      %
% typeOfData: string that defines the type of input data matrix; 'im'   %
% for images and 'ge' for gene expression data. This input argument is  %
% necessary, since in the case of image data the nodes in the output    %
% layer should not be linear                                            %
% initCellVector: cell array including the weights and biases for the   %
% PCA-based initialization. In case of random-based initialization an   %
% empty row vector should be used (i.e., [])                            %
% max_epochs: maximum number of training iterations for the newly       %
% created network                                                       %
% varargin: optional user-defined argument for cross validation. If     %
% this parameter is set to 'cv', then the default division function     %
%'dividerand' is replaced by 'dividetrain', since during cross          %
% validation the division of data is invoked by the respective          % 
% splitting of the user's implementation.                               %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% net: network object with appropriately initialized weights and biases %
% as well as some other training parameters. The returned network object%
% can be used for training and testing.                                 %
%                                                                       %
%                                                                       %
% IMPORTANT !!!                                                         %
% initCellVector should be a 1-by-m cell array, where m is the number of%
% hidden plus output layers of the network multiplied by 2. The first   %
% m/2 cells contain the weights in increasing order, while the second   %
% m/2 cells contain the biases in increasing order.                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function net = customizeNetwork(architecture, activationFunction, typeOfData, initCellVector, max_epochs, varargin)


if (nargin < 5)
    
    error('Invalid number of inputs\n');
        
elseif (length(varargin) > 1)
    
    error('Invalid number of optional input arguments\n');
    
elseif ((length(varargin) == 1) && (strcmpi(varargin{1}, 'cv') ~= 1))
    
    error('Invalid type of optional input arguments\n');
    
elseif ((strcmpi(typeOfData, 'im') ~= 1) && (strcmpi(typeOfData, 'other') ~= 1))
    
    error('Invalid type of data\n');
        
else
    
    if (strcmpi(activationFunction, 'logsig') == 1)

        mode = 1;

    elseif (strcmpi(activationFunction, 'tansig') == 1)

        mode = 2;

    else

        error('Invalid type of activation function. Choose between "logsig" and "tansig"\n');

    end

    
    %% common settings despite the network structure
    
    % length of vector with the network structure
    len = length(architecture);
    
    % network with 1 input and (len - 1) layers
    net = network(1, (len - 1));
    
    % length of row vector with initialized weights and biases
    initLen = length(initCellVector);
    
    % set to 1 all layers that have biases
    net.biasConnect = ones((len - 1), 1);
    
    % only 1st layer has weights that come from inputs
    net.inputConnect(1, 1) = 1;
    
    %% different settings based on the network structure
    
    if (initLen == 4)               % pca initialization with 1 hidden layer (3 layers in total)
       
        net.layerConnect(2, 1) = 1;
        
        % layer that generate outputs
        net.outputConnect(1, 2) = 1;
        
        %% customize layers
        
        % dimensions of input vectors
        net.inputs{1}.size = architecture(1);
        
        % 1st hidden layer
        net.layers{1}.dimensions = architecture(2);
        net.layers{1}.name = 'Hidden 1';
        
        % output layer
        net.layers{2}.dimensions = architecture(3);
        net.layers{2}.name = 'Output';
        
        % user-defined activation functions
        if (mode == 1)

            net.layers{1}.transferFcn = 'logsig';

            if (strcmpi(typeOfData, 'im') == 1)

                net.layers{2}.transferFcn = 'logsig';

            end

        else

            net.layers{1}.transferFcn = 'tansig';

            if (strcmpi(typeOfData, 'im') == 1)

                net.layers{2}.transferFcn = 'tansig';

            end

        end
        
        %% weights & biases initialization
        
         % 1st hidden layer
        net.IW{1, 1} = initCellVector{1, 1};

        % output layer
        net.LW{2, 1} = initCellVector{1, 2};
        
        % 1st hidden layer
        net.b{1, 1} = initCellVector{1, 3}(:, 1);

        % output layer
        net.b{2, 1} = initCellVector{1, 4}(:, 1);
        
    elseif (initLen == 6)              % pca initialization with 2 hidden layers (4 layers in total)
        
        net.layerConnect(2, 1) = 1;
        net.layerConnect(3, 2) = 1;
        
        % layer that generate outputs
        net.outputConnect(1, 3) = 1;
        
        %% customize layers
        
        % dimensions of input vectors
        net.inputs{1}.size = architecture(1);
        
        % 1st hidden layer
        net.layers{1}.dimensions = architecture(2);
        net.layers{1}.name = 'Hidden 1';
        
        % 2nd hidden layer
        net.layers{2}.dimensions = architecture(3);
        net.layers{2}.name = 'Hidden 2';
        
        % output layer
        net.layers{3}.dimensions = architecture(4);
        net.layers{3}.name = 'Output';
        
        % user-defined activation functions
        if (mode == 1)

            net.layers{1}.transferFcn = 'logsig';
            net.layers{2}.transferFcn = 'logsig';

            if (strcmpi(typeOfData, 'im') == 1)

                net.layers{3}.transferFcn = 'logsig';

            end

        else

            net.layers{1}.transferFcn = 'tansig';
            net.layers{2}.transferFcn = 'tansig';

            if (strcmpi(typeOfData, 'im') == 1)

                net.layers{3}.transferFcn = 'tansig';

            end

        end
        
        %% weights & biases initialization
        
        % 1st hidden layer
        net.IW{1, 1} = initCellVector{1, 1};

        % 2nd hidden layer
        net.LW{2, 1} = initCellVector{1, 2};

        % output layer
        net.LW{3, 2} = initCellVector{1, 3};
        
        % 1st hidden layer
        net.b{1, 1} = initCellVector{1, 4}(:, 1);

        % 2nd hidden layer
        net.b{2, 1} = initCellVector{1, 5}(:, 1);

        % output hidden layer
        net.b{3, 1} = initCellVector{1, 6}(:, 1);
        
    elseif (initLen == 8)            % pca initialization with 3 hidden layers (5 layer in total)
        
        net.layerConnect(2, 1) = 1;
        net.layerConnect(3, 2) = 1;
        net.layerConnect(4, 3) = 1;

        % layer that generate outputs
        net.outputConnect(1, 4) = 1;

        %% customize layers

        % dimensions of input vectors
        net.inputs{1}.size = architecture(1);

        % 1st hidden layer
        net.layers{1}.dimensions = architecture(2);
        net.layers{1}.name = 'Hidden 1';

        % 2nd hidden layer
        net.layers{2}.dimensions = architecture(3);
        net.layers{2}.name = 'Hidden 2';

        % 3rd hidden layer
        net.layers{3}.dimensions = architecture(4);
        net.layers{3}.name = 'Hidden 3';

        % output layer
        net.layers{4}.dimensions = architecture(5);
        net.layers{4}.name = 'Output';

        % user-defined activation functions
        if (mode == 1)

            net.layers{1}.transferFcn = 'logsig';
            net.layers{2}.transferFcn = 'logsig';
            net.layers{3}.transferFcn = 'logsig';

            if (strcmpi(typeOfData, 'im') == 1)

                net.layers{4}.transferFcn = 'logsig';

            end

        else

            net.layers{1}.transferFcn = 'tansig';
            net.layers{2}.transferFcn = 'tansig';
            net.layers{3}.transferFcn = 'tansig';

            if (strcmpi(typeOfData, 'im') == 1)

                net.layers{4}.transferFcn = 'tansig';

            end

        end
        
        %% weights & biases initialization
        
        % 1st hidden layer
        net.IW{1, 1} = initCellVector{1, 1};

        % 2nd hidden layer
        net.LW{2, 1} = initCellVector{1, 2};

        % 3rd hidden layer
        net.LW{3, 2} = initCellVector{1, 3};
        
        % output layer
        net.LW{4, 3} = initCellVector{1, 4};
        
        % 1st hidden layer
        net.b{1, 1} = initCellVector{1, 5}(:, 1);

        % 2nd hidden layer
        net.b{2, 1} = initCellVector{1, 6}(:, 1);

        % 3rd hidden layer
        net.b{3, 1} = initCellVector{1, 7}(:, 1);
        
        % output layer
        net.b{4, 1} = initCellVector{1, 8}(:, 1);
              
    else
        
        if (len == 3)               % random initialization with 1 hidden layer (3 layers in total)
            
            net.layerConnect(2, 1) = 1;
        
            % layer that generate outputs
            net.outputConnect(1, 2) = 1;

            %% customize layers

            % dimensions of input vectors
            net.inputs{1}.size = architecture(1);

            % 1st hidden layer
            net.layers{1}.dimensions = architecture(2);
            net.layers{1}.name = 'Hidden 1';

            % output layer
            net.layers{2}.dimensions = architecture(3);
            net.layers{2}.name = 'Output';

            % user-defined activation functions
            if (mode == 1)

                net.layers{1}.transferFcn = 'logsig';

                if (strcmpi(typeOfData, 'im') == 1)

                    net.layers{2}.transferFcn = 'logsig';

                end

            else

                net.layers{1}.transferFcn = 'tansig';

                if (strcmpi(typeOfData, 'im') == 1)

                    net.layers{2}.transferFcn = 'tansig';

                end

            end
              
            
        elseif (len == 4)              % random initialization with 2 hidden layers (4 layers in total)
            
            net.layerConnect(2, 1) = 1;
            net.layerConnect(3, 2) = 1;

            % layer that generate outputs
            net.outputConnect(1, 3) = 1;

            %% customize layers

            % dimensions of input vectors
            net.inputs{1}.size = architecture(1);

            % 1st hidden layer
            net.layers{1}.dimensions = architecture(2);
            net.layers{1}.name = 'Hidden 1';

            % 2nd hidden layer
            net.layers{2}.dimensions = architecture(3);
            net.layers{2}.name = 'Hidden 2';

            % output layer
            net.layers{3}.dimensions = architecture(4);
            net.layers{3}.name = 'Output';

            % user-defined activation functions
            if (mode == 1)

                net.layers{1}.transferFcn = 'logsig';
                net.layers{2}.transferFcn = 'logsig';

                if (strcmpi(typeOfData, 'im') == 1)

                    net.layers{3}.transferFcn = 'logsig';

                end

            else

                net.layers{1}.transferFcn = 'tansig';
                net.layers{2}.transferFcn = 'tansig';

                if (strcmpi(typeOfData, 'im') == 1)

                    net.layers{3}.transferFcn = 'tansig';

                end

            end
            
        elseif (len == 5)      % random initialization with 3 hidden layers (5 layers in total)         
            
            net.layerConnect(2, 1) = 1;
            net.layerConnect(3, 2) = 1;
            net.layerConnect(4, 3) = 1;
            
            % layer that generate outputs
            net.outputConnect(1, 4) = 1;
            
            %% customize layers

            % dimensions of input vectors
            net.inputs{1}.size = architecture(1);

            % 1st hidden layer
            net.layers{1}.dimensions = architecture(2);
            net.layers{1}.name = 'Hidden 1';

            % 2nd hidden layer
            net.layers{2}.dimensions = architecture(3);
            net.layers{2}.name = 'Hidden 2';

            % 3rd hidden layer
            net.layers{3}.dimensions = architecture(4);
            net.layers{3}.name = 'Hidden 3';
            
            % output layer
            net.layers{4}.dimensions = architecture(5);
            net.layers{4}.name = 'Output';
            
            % user-defined activation functions
            if (mode == 1)

                net.layers{1}.transferFcn = 'logsig';
                net.layers{2}.transferFcn = 'logsig';
                net.layers{3}.transferFcn = 'logsig';
                
                if (strcmpi(typeOfData, 'im') == 1)

                    net.layers{4}.transferFcn = 'logsig';

                end

            else

                net.layers{1}.transferFcn = 'tansig';
                net.layers{2}.transferFcn = 'tansig';
                net.layers{3}.transferFcn = 'tansig';
                
                if (strcmpi(typeOfData, 'im') == 1)

                    net.layers{4}.transferFcn = 'tansig';

                end

            end
            
            
        else
            
            error('Invalid network structure\n');
                 
        end
    
        
    end
    

    %% other parameters

    if (length(varargin) == 1)

        % use all training data from K-1 folds in CV inner loop
        net.divideFcn = 'dividetrain';

    else

        % randomly division to training, validation and test sets of the training
        % dataset
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;

    end

    % training function
    net.trainFcn = 'trainscg';
    net.trainParam.epochs = max_epochs;
    
end

end