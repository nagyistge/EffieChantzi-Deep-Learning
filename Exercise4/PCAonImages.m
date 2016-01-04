        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercises 3, 4 %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function performs PCA on data given by the user, based on the    %
% built-in matlab implementation 'pca'.                                 %
% Apart from the standard mathworks functionality of 'pca', two more    %
% options are provided by this customized function. Firstly, the        %
% reconstruction of the compressed data in the user-defined 'M' reduced %   
% dimensions is obtained, and secondly, the total mean square error     %
% between the whole set of original and reconstructed data is estimated.%
% This two additional options are necessary for the fulfillment of      %
% Exercise 3.                                                           %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% data: data input matrix. Rows correspond to dimensions and columns    %
% correspond to observations.                                           %
% PCs_M: user-defined 'M' principal components, for the calculation of  %
% the reconstruction of the compressed data back to initial number of   %   
% dimensions and the total mean square error                            %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% coeff: PCs_M user-defined principal component coefficients in         %
% decreasing order of component variance                                %
% score: PCs_M user-defined principal component scores, meaning the     %
% representation of input data in the principle component space. Rows   %
% of score correspond to observations and rows correspond to components.%
% latent: PCs_M user-defined principal component variances, meaning the %
% eigenvalues in decreasing order                                       %
% mu: estimated mean of each variable in input data                     %                                                                      
% totalVarPCs_M: total variance covered by the PCs_M user-defined       %
% principal components                                                  %
% reconstructedImages: reconstruction of the compressed input data to   %
% the user-defined PCs_M principal components                           %
% totalMSE: total mean square error between the original input data and %
% the reconstructed data after the compression to user-defined PCs_M    %
% principal components                                                  %
%                                                                       %
% It is possible to skip any of the output arguments by putting the     %
% tilda symbol (~), on these ones that you do not wish to be returned   %
% as outputs.                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [coeff, score, latent, mu, totalVarPCs_M, reconstructedImages, totalMSE] = PCAonImages(data, PCs_M)

% number of observations
observations = size(data, 2);

% built-in pca 
[coeff, score, latent, ~, explained, mu] = pca(data');

% outputs adjusted only to PCs_M user-defined principal components
coeff = coeff(:, (1 : PCs_M));
score = score(:, (1 : PCs_M));
latent = latent(1 : PCs_M);

% total variance covered by the user-defined 'PCs_M' principle components
totalVarPCs_M = sum(explained((1 : PCs_M), 1));

%% Reconstruction of compressed input data

mu = repmat(mu', 1, observations);
reconstructedImages = coeff*score' + mu;

%% Total Mean Square Error on the whole input dataset
totalMSE = estimateTotalMSE(data, reconstructedImages);

end

