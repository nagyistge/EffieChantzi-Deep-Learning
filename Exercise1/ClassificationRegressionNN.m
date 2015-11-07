%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Chantzi Efthymia - Deep Learning - Exercise 1  %%
%%                      Task A                    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function performs either a classification or a regression task   %
% using the appropriate datasets and 'nntraintool' of Matlab2015b       %
% Neural Network Toolbox.                                               %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
%                                                                       %
% inputs: training data                                                 %
% targets: target output data                                           %                 
% task: char specifying task. There are two options:                    %
% i) 'C'/'c' -> classification task                                     %
% ii) 'R'/'r' -> regression                                             %        
%                                                                       %
% varargin: optional set of inputs defined by user as a cell array      % 
% There are three options, which can also be combined:                  %                             
% i) {'neurons', positive integer} -> neurons in hidden layer           %
% ii) {'division', [trainPer valPer testPer]} -> random division into   %
% training, validation and test sets. The amount of division is         %
% described by the row vector [trainPer valPer testPer]. These values   %
% should be in the range [0, 1] and add up to 1.                        %
% iii) {'trainFunction', nameOfTrainFunction} -> algorithm used in the  %
% network training (i.e., 'trainlm')                                    %
% If varargin is not provided by the user, the following default values %                                
% are set:                                                              %
% hiddenLayerSize = 10                                                  %
% trainPer = 0.7                                                        %
% valPer = 0.15                                                         %
% testPer = 0.15                                                        %
% net.trainFcn = 'trainlm' -> regression/function fitting               %
% net.trainFcn = 'trainscg' -> classification/pattern recognition       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
%                                                                       %
% GUI of 'nntraintool'. Plots of performance measures are provided.     %
%                                                                       %
%                                                                       %
% IMPORTANT!!                                                           %
% Choose the appropriate dataset for each task. Type 'help nndatasets'  % 
% in the command line prompt to have access to all the built-in neural  %
% network datasets that are suitable for the different cases. Depending %
% on your preference, just do '[inputs, targets] = name_dataset;'       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function  ClassificationRegressionNN(inputs, targets, task, varargin)

if ((nargin < 3) || (nargin > 4))
   
    error('Invalid number of inputs');
    
else
    
    hiddenLayerSize = 10;
    trainPer = 0.7;
    valPer = 0.15;
    testPer = 0.15;
    tF = '';
    
    if(nargin == 4)
        
        options = varargin{:};

        if ((mod(length(options), 2) ~= 0) || (length(options) > 6))

            error('Invalid optional arguments');

        else

           for i = 1 : 2 : (length(options) - 1)

               if (strcmpi(options{i}, 'neurons') == 1)

                   hiddenLayerSize = options{i + 1};

               elseif (strcmpi(options{i}, 'trainFunction') == 1)

                   tF = options{i + 1};

               elseif (strcmpi(options{i}, 'division') == 1)

                   if (length(options{i + 1}) < 3)

                       error('Invalid vector for division of data');

                   else

                       trainPer = options{i + 1}(1);
                       valPer = options{i + 1}(2);
                       testPer = options{i + 1}(3);

                   end

               else

                   error('Invalid format of optional input argument');

               end

          end

        end
       
    end
        
end

%% Function fitting (Regression) or Pattern Recognition (Classification)

if (strcmpi(task, 'R') == 1) % regression task
    
    net = fitnet(hiddenLayerSize); 
    
elseif (strcmpi(task, 'C') == 1) % classification task
    
    net = patternnet(hiddenLayerSize);

else
    
    error('Invalid/Missing operation'); 
    
end
net.divideParam.trainRatio = trainPer;
net.divideParam.valRatio = valPer;
net.divideParam.testRatio = testPer;

%% Training of Neural Network
if(~isempty(tF))
    
    net.trainFcn = tF;
        
end
train(net, inputs, targets);

end