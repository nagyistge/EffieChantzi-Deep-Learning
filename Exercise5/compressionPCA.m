        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise 5 %%
        %%                  Tasks A, C                   %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script pertains to the implementation of tasks A and C, where PCA %
% is applied to the training set of data with the goal of compression    % 
% from the original number of dimensions to 'L' user-defined reduced     %
% dimensions, while maintaining at the same time a specific total mean   %
% reconstruction error.                                                  %
% This script is specifically adjusted to two different datasets;        % 
% time-lapse microscopy movies(images_40.mat) and mRNA expression        %
% profiles from glioma cell lines(glio_mRNA_data.mat). Both datasets     %
% have been retrieved, processed and saved in .mat format for the needs  %
% of Exercise 5.                                                         %
%                                                                        %
% Run this script and a menu will guide you through. More precisely, the % 
% dataset and the number of the 'L' latent variables used for compression%
% are requested. In case the time-lapse movie dataset is selected, the   %
% indices of the subset of images for visual inspection are also         %
% requested. Results and a figure(for the movie dataset) are displayed.  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc;
close all;
clear all;


fprintf('--------------------------- PCA Data Compression --------------------------\n');

%% Request user-defined data selection (either time-lapse movies or mRNA gene expression data)

dataType = '';
while ((str2double(dataType) ~= 1) && (str2double(dataType) ~= 2))
   
    fprintf('Select data type:\n');
    fprintf('1. Time-lapse microscopy movies\n');
    fprintf('2. Conventional Glioma mRNA\n');
    
    dataType = input('', 's');
    
end
dataType = str2double(dataType);

%% Load appropriate dataset into memory

if (dataType == 1)
   
  load 'images_40.mat';
  data = images_40;  
  reshapeVector = [40 40];
  mode = 'im';
  
else
    
    load 'glio_mRNA_data.mat';
    data = glio_mRNA_data;
    mode = 'ge';
    
end

%% Request user-defined reduced dimensions

% principal components of the M reduced dimensions
PCs_M = 0;
while ((PCs_M <= 0) || (mod(PCs_M, 1) ~= 0))

    PCs_M = input('Enter a positive value for the first M Principal Components(reduced dimensions): \n');
    
end

%% PCA on input data

[coeff, score, latent, mu, totalVarPCs_M, reconstructions, totalMSE] = PCAonData(data, PCs_M, mode);


if (dataType == 1)
   
    %% Visual inspection of user-defined(randomly) original vs. reconstructed images
    fprintf('--------------------------- Visual Inspection after Compression --------------------------\n');
    fprintf('------------- Indexes of a subset of images -------------\n\n')
    firstIm = 0;
    while ((firstIm <= 0) || (mod(firstIm, 1) ~= 0))

        firstIm = input('Give the index of the first image: \n');

    end

    lastIm = 0;
    while ((lastIm <= 0) || (mod(lastIm, 1) ~= 0))

        lastIm = input('Give the index of the last image: \n');

    end
    fprintf('------------------------------------------------------------------------------------------\n\n');


    % visual inspection of a subset of original vs. reconstruction images
    plotComparison(data, reconstructions, [firstIm lastIm], [], reshapeVector, totalMSE);
    
else
    
    fprintf('--------------------------- Inspection after Compression --------------------------\n');
    fprintf('mRNA gene expression average of relative reconstruction errors: %.5f\n', totalMSE);
    
end

