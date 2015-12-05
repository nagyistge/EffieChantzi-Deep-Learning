        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise  5 %%
        %%                    Tasks A, C                  %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script pertains to the implementation of task A and C, where PCA  %
% is used for weights and biases initialization of a 3-hidden layer deep % 
% network trained for image and mRNA gene expression data reconstruction.% 
% This script is specifically adjusted to two different datasets;        % 
% time-lapse microscopy movies('images_40.mat') and mRNA expression      %
% profiles from glioma cell lines('glio_mRNA_data.mat'). Both datasets   %
% have been retrieved, processed and saved in .mat format for the needs  %
% of Exercise 5.                                                         %
%                                                                        %
%                                                                        %
% Run this script and a menu will guide you through. More precisely, the % 
% dataset, type of input dataset(compresssed or uncompressed), the size  %
% of the 3 hidden layers as a row vector, as well as the type of the     %
% neurons' activation function(logsig/tansig) are requested.             %
%                                                                        %
% In case the time-lapse movie dataset is selected, the performance is   %
% evaluated both on the training set and on the independent test set     %
% ('test_images_40.mat'). The total mean reconstruction error is         %
% accompanied with a plot of 5 original vs. their respective             %
% reconstructed images for visual inspection.                            %
% For the mRNA gene expression dataset('glio_mRNA_data.mat'), the average%
% relative error is estimated only on the training set and the results   %
% are printed on the command prompt.                                     %
%                                                                        %
% IMPORTANT !!!                                                          %
% Here, s = 1/40 (line 55).                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
close all;
clear all;

fprintf('--------------------------- Deep Bottleneck Network with PCA initialization --------------------------\n');


%% Request user-defined data selection (either time-lapse movies or mRNA gene expression data)

dataType = '';
while ((str2double(dataType) ~= 1) && (str2double(dataType) ~= 2))
   
    fprintf('Select data type:\n');
    fprintf('1. Time-lapse microscopy movies\n');
    fprintf('2. Conventional Glioma mRNA\n');
    
    dataType = input('', 's');
    
end
dataType = str2double(dataType);


%% Request used-defined type of inputs (either compressed after PCA or uncompressed/original)
inputType = '';
while ((str2double(inputType) ~= 1) && (str2double(inputType) ~= 2))
   
    fprintf('Select type of inputs:\n');
    fprintf('1. Compressed with PCA\n');
    fprintf('2. Uncompressed\n');
    
    inputType = input('', 's');
    
end
inputType = str2double(inputType);

%% Load appropriate dataset into memory

if (dataType == 1)
   
  cd data
  load 'images_40.mat';
  cd ..
  data = images_40;  
  reshapeVector = [40 40];
  mode = 'im';
  
else
    
    cd data
    load 'glio_mRNA_data.mat';
    cd ..
    data = glio_mRNA_data;
    mode = 'ge';
    
end

if (inputType == 1)
   
    %% Request user-defined reduced dimensions

    % principal components of the M reduced dimensions
    PCs_M = '';
    while ((str2double(PCs_M) <= 0) || (mod(str2double(PCs_M), 1) ~= 0))

        PCs_M = input('Enter a positive value for the first M Principal Components(reduced dimensions): \n', 's');
    
    end
    PCs_M = str2double(PCs_M);
     
    %% PCA on training data

    [coeff, score, latent, mu, totalVarPCs_M, reconstructions, totalMSE] = PCAonData(data, PCs_M, mode);
    
    % compressed inputs
    inputs = score(:, (1 : PCs_M))';
    
else
    
    % uncompresssed inputs
    inputs = data;
    
end
dim = size(inputs, 1);

%% Request user-defined hidden layer sizes

hiddenSize = 0;
while ((sum(mod(hiddenSize, 1)) > 0) || (sum(hiddenSize < 1) > 0) || (length(hiddenSize) ~= 3) || (isempty(hiddenSize)))

    hiddenSize = input('Enter the size of the 3 hidden layers as a row vector: \n');
    
end

activationFunction = '';
while ((~strcmpi(activationFunction, 'logsig') && ~strcmpi(activationFunction, 'tansig')) || (isempty(activationFunction)))
    
    activationFunction = input('Enter the name of the transfer function to be used (logsig/tansig):\n', 's');
    
end

%% PCA Initialization

[W1, W2, W3, W4, b1, b2, b3, b4] = PCAInitialization([hiddenSize(1) hiddenSize(2)], 1/40, inputs, activationFunction, mode);

%% Weights and biases initialization 

architectureVector = [dim hiddenSize dim];
net = customizeNetwork(architectureVector, W1, W2, W3, W4, b1, b2, b3, b4, activationFunction, mode);

%% Training

net.trainParam.epochs = 1500;
deepnet = train(net, inputs, inputs);

reconstructions_training = deepnet(inputs);

% compressed inputs
if (inputType == 1)
    
    
    reconstructions_training = coeff(:, (1 : PCs_M))*reconstructions_training + mu;
    totalMSE_PCA_training = estimateTotalMSE(data, reconstructions_training, mode);
    
    % time-lapse movie dataset
    if(dataType == 1)
   
        plotComparison(data, reconstructions_training, [1 5], [PCs_M hiddenSize PCs_M], reshapeVector, totalMSE_PCA_training);
        plotComparison(data, reconstructions, [1 5], [], reshapeVector, totalMSE);
        
    % mRNA gene expression dataset    
    else
       
        fprintf('--------------------------- Inspection after Compression --------------------------\n');
        fprintf('--- mRNA gene expression average of relative reconstruction errors ---\n');
        fprintf('Only compression with PCA to %d dimensions: %.5f\n', PCs_M, totalMSE);
        fprintf('Further compression with bottleneck SAE initialization: %.5f\n', totalMSE_PCA_training);
        
    end

% original/uncompressed inputs
else
    
    totalMSE_PCA_training = estimateTotalMSE(inputs, reconstructions_training, mode);
    
    % time-lapse movie dataset
    if(dataType == 1)
       
        plotComparison(inputs, reconstructions_training, [1 5], architectureVector, reshapeVector, totalMSE_PCA_training);
        
    else
        
        fprintf('--- mRNA gene expression average of relative reconstruction errors ---\n');
        fprintf('Bottleneck PCA initialization: %.5f\n', totalMSE_PCA_training);
        
    end
    
end

%% Test trained network on test dataset only in case of time-lapse movies
if (dataType == 1)
    
    %% Load Test dataset into memory

    cd data
    load 'test_images_40.mat';
    cd ..
    
    outputs = deepnet(test_images_40);

    % total reconstruction error
    fprintf('\n');
    fprintf('-------------------- Testing Performance --------------------\n');
    totalMSE_PCA_test = estimateTotalMSE(test_images_40, outputs, 'im');
    fprintf('Total MSE on the test dataset: %.5f', totalMSE_PCA_test);
    fprintf('\n');

    % visual inspection of first 5 original vs. reconstructed test images
    plotComparison(test_images_40, outputs, [1 5], architectureVector, reshapeVector, totalMSE_PCA_test);
    
end

