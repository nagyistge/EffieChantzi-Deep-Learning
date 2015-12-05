        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercises 3,4,5 %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function performs PCA on data given by the user, based on the    %
% built-in matlab implementation 'pca'.                                 %
% Apart from the standard mathworks functionality of 'pca', two more    %
% options are provided by this customized function. Firstly, the        %
% reconstruction of the compressed data in the user-defined 'L' reduced %   
% dimensions is obtained, and secondly, the total mean square error     %
% between the whole set of original and reconstructed data is estimated.%
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% data: data input matrix. Rows correspond to observations and columns  %
% correspond to variables, meaning the dimensions.                      %
% PCs_M: user-defined 'L' latent variables, for the calculation of the  %
% reconstruction of the compressed back to the uncompressed data and    %   
% the total mean squared error of reconstruction.                       %
% typeOfData: string that defines the type of input data matrix; 'im'   %
% for images and 'ge' for gene expression data. This input argument is  %
% necessary, since the formula for the estimation of the mean squared   %
% error differs between images and gene expression data.                %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% coeff: principle component coefficients in decreasing order of        %
% component variance                                                    %
% score: principle component scores, meaning the representation of input%
% data in the principle component space. Rows of score correspond to    %
% observations and rows correspond to components.                       %
% latent: principle component variances, meaning the eigenvalues in     %
% decreasing order                                                      %
% mu: estimated mean of each variable in input data                     %                                                                      
% totalVarPCs_M: total variance covered by the user-defined 'M'         %
% principle components                                                  %
% reconstructedImages: reconstruction of the compressed input data to   %
% the user-defined 'M' principal components                             %
% totalMSE: total mean square error between the original input data and %
% the reconstructed data after the compression to user-defined 'M'      %
% principal components                                                  %
%                                                                       %
% It is possible to skip any of the output arguments by putting the     %
% tilda symbol (~), on these ones that you do not wish to be returned   %
% as outputs.                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [coeff, score, latent, mu, totalVarPCs_M, reconstructions, totalMSE] = PCAonData(data, PCs_M, typeOfData)

if ((strcmpi(typeOfData, 'im') ~= 1) && (strcmpi(typeOfData, 'ge') ~= 1))
    
    error('Invalid type of input data');
    
else
    
    % number of observations
    observations = size(data, 2);

    [coeff, score, latent, ~, explained, mu] = pca(data');     

    % total variance covered by the user-defined 'PCs_M' principle components
    totalVarPCs_M = sum(explained((1 : PCs_M), 1));

    %% Reconstruction of compressed input data

    mu = repmat(mu', 1, observations);
    reconstructions = coeff(:, (1 : PCs_M))*score(:, (1 : PCs_M))' + mu;

    %% Total Mean Square Error on the whole input dataset
    totalMSE = estimateTotalMSE(data, reconstructions, typeOfData); 
    
   
end

end

