        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise 2  %%
        %%                    Tasks A, B                  %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function trains and tests a classifier on training and test data,%
% respectively. The classifier is a feedforward neural network with one %
% or more hidden layers. The goal is to classify digit images as        %
% as belonging to class "3" or "other". It uses the functions:          %
% "createInputs", "createNewLabels", "trainPatternnet" and              %
% "testPatternnet".                                                     %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% XTrain: matrix-formed training data                                   %
% LTrain: matrix-formed labels for XTrain (training data)               %
% XTest: matrix-formed test data                                        %
% LTest: matrix-formed labels for XTest (test data)                     %
% N: row vector of first and last used image as indexes                 %
% hiddenSize: number of neurons in for one hidden layer and a row       %
% vector for multiple hidden layers.                                    %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% probabilities:  (%) false alarm and detection probabilities on the    %
% test dataset (XTest, LTest)                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, N, hiddenSize, varargin)

if (nargin < 6)
    
    error('Invalid number of input arguments');
    
else

    %% Train Pattern Recognition Neural Network 
    
    images = createInputs(XTrain, N);
    labels = createNewLabels(LTrain, N);
    
    if(isempty(varargin))
        
        net = trainPatternnet(hiddenSize, images, labels);
        
    elseif ((length(varargin) == 1) && (strcmpi(varargin{1}, 'mse') == 1))
        
        net = trainPatternnet(hiddenSize, images, labels, 'mse');
        
    else
       
        error('Invalid number of optional input arguments');
        
    end
    
    %% Test Pattern Recognition Neural Network test dataset
    
    testImages = createInputs(XTest, N);
    testLabels = createNewLabels(LTest, N);

    % false alarm and detection probabilities on the test dataset   
    probabilities = testPatternnet(net, testImages, testLabels);
    
end

end