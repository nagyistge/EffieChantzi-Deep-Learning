        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise 3  %%
        %%                    Task A                      %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script pertains to the implementation of task A, where PCA is    %
% applied to the training set of images (digittrain_dataset) with the   %
% goal of compression from 784 to 'M' user-defined dimensions with a    %
% visually acceptable result of reconstruction.                         %
% This visual result is tested on a subset of user-defined training     %
% images, while the total mean square error of the reconstruction on    %
% the whole dataset is calculated and displayed on the title of the     %
% respective figure.                                                    %
%                                                                       %
%                                                                       %
% Run this script and a menu will guide you through. More precisely,    % 
% the number of the 'M' principle components used for compression as    %
% well as the indexes of the subset of images for visual inspection are %
% asked. Results and a figure for the necessary visual inspection are   %
% displayed.                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
close all;
clear all;

%% Load trainning dataset into memory

[XTrain, LTrain] = digittrain_dataset;

% transform the cell array of XTrain images into a matrix of vectors taking
% only the 5 first images
images = createInputs(XTrain, [1 5000]);

% create labels for "3" and "others" classes
labels = createNewLabels(LTrain, [1 5000]);

fprintf('--------------------------- PCA Image Compression --------------------------\n');

%% Request user-defined reduced dimensions

% principal components of the M reduced dimensions
PCs_M = 0;
while ((PCs_M <= 0) || (mod(PCs_M, 1)~= 0))

    PCs_M = input('Enter a positive value for the first M Principal Components(reduced dimensions): \n');
    
end

%% PCA on training data

[~, ~, ~, ~, ~, reconstructedImages, totalMSE] = PCAonImages(images, PCs_M);

%% Visual inspection of user-defined(randomly) original vs. reconstructed images
fprintf('--------------------------- Visual Inspection after Compression --------------------------\n');
fprintf('------------- Indexes of a subset of images -------------\n\n')
firstIm = 0;
while ((firstIm <= 0) || (mod(firstIm, 1)~= 0))

    firstIm = input('Give the index of the first image: \n');
    
end

lastIm = 0;
while ((lastIm <= 0) || (mod(lastIm, 1)~= 0))

    lastIm = input('Give the index of the last image: \n');
    
end
fprintf('------------------------------------------------------------------------------------------\n\n');


% visual inspection of a small set of original vs. reconstruction images
plotComparison(images, reconstructedImages, [firstIm lastIm], [], totalMSE);


temp = zeros(1, 5000);
for i = 1 : 5000
    
    temp(1, i) = (sum((images(:, i) - reconstructedImages(:, i)).^2))/784;
    
end
manualMean = mean(temp);