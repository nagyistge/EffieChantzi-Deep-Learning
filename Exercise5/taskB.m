        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise  5 %%
        %%                     Task B                     %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script pertains to the implementation of task B, where the initial% 
% image dataset('images_40.mat') from the time-lapse microscopy movies   %
% is randomly divided into two equally large halves. PCA is applied to   %
% the original dataset as well as to one of the two subsets. The number  % 
% of latent variables that results in a similar error to that one of PCA %
% in the whole dataset, is selected and used for compression and         % 
% reconstruction to the other subset. In this way, overfitting is        %
% avoided. The goal is to find the number of latent variables from the   %
% one subset that are adequate for the other unseen subset, while getting%
% similar errors to the that one from PCA on the whole dataset.          %
%                                                                        %
% Run this script and a menu will guide you through. More precisely, the % 
% number of latent vriables to be used, are requested.                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;
close all;

cd data
load 'images_40.mat';
cd ..
reshapeVector = [40 40];

% total number of observations
observations = size(images_40, 2);

% random indices for two equally sized halves
ind = crossvalind('Kfold', observations, 2);

% indices that respond to first and second half respectively
indices_subset1 = (ind == 1);
indices_subset2 = ~indices_subset1;

% create the two equally sized subsets
subset1 = images_40(:, indices_subset1);
subset2 = images_40(:, indices_subset2);

fprintf('--------------------------- PCA Data Compression --------------------------\n\n');
fprintf('--- Original Dataset ---\n');
[coeff, score, latent, mu, totalVarPCs_M, reconstructions, totalMSE] = PCAonData(images_40, 50, 'im');
plotComparison(images_40, reconstructions, [7 11], [], reshapeVector, totalMSE);
fprintf('Total MSE: %.5f\n', totalMSE);
fprintf('\n');

%% Request user-defined reduced dimensions for the 1st half

% principal components of the M reduced dimensions
PCs_M = 0;
while ((PCs_M <= 0) || (mod(PCs_M, 1) ~= 0))

    PCs_M = input('Enter a positive value for the first M Principal Components(reduced dimensions): \n');
    
end
[coeff1, score1, latent1, mu1, totalVarPCs_M1, reconstructions1, totalMSE1] = PCAonData(subset1, PCs_M, 'im');
plotComparison(subset1, reconstructions1, [300 304], [], reshapeVector, totalMSE1);
fprintf('--- First Half ---\n');
fprintf('Total MSE: %.5f\n', totalMSE1);
fprintf('\n');

% test the number of latent variables from the 1st half to the 2nd
% independent half
[coeff2, score2, latent2, mu2, totalVarPCs_M2, reconstructions2, totalMSE2] = PCAonData(subset2, PCs_M, 'im');
plotComparison(subset2, reconstructions2, [300 304], [], reshapeVector, totalMSE2);
fprintf('--- Second Half ---\n');
fprintf('Total MSE: %.5f\n', totalMSE2);
fprintf('\n');
