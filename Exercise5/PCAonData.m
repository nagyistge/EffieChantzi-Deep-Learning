           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %% Chantzi Efthymia - Deep Learning - Exercise 5 %%
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function performs PCA on data given by the user, based on the    %
% built-in matlab implementation 'pca'.                                 %
% Apart from the standard mathworks functionality of 'pca', two more    %
% options are provided by this customized function. Firstly, the        %
% reconstruction of the compressed data in the PCs_M user-defined       %   
% dimensions is obtained, and secondly, the total mean squared error    %
% between the whole set of original and reconstructed data is estimated.%
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% data: data input matrix. Rows correspond to variables, meaning the    %
% dimensions and columns correspond to observations.                    %
% PCs_M: user-defined 'L' latent variables, for the calculation of the  %
% reconstruction of the compressed back to the uncompressed data and    %   
% the total mean squared error of reconstruction.                       %
% typeOfData: string that defines the type of input data matrix; 'im'   %
% for images/other and 'ge' for gene expression data. This input        %
% argument is necessary, since the formula for the estimation of the    %
% mean squared error differs for gene expression data.                  %
% varargin: optional user-defined argument for root mean squared error. %
% If this parameter is set to 'rmse', then apart from the total mean    %
% squared error, the total root mean squared error is estimated and     %
% returned as an optional output argument(varargout).                   %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% coeff: PCs_M principal component coefficients in decreasing order of  %
% component variance                                                    %
% score: PCs_M user-defined principal component scores, meaning the     %
% representation of input data in the principle component space. Rows   %
% correspond to observations and rows to components.                    %
% latent: PCs_M userd-defined principal component variances, meaning    %
% the eigenvalues in  decreasing order                                  %
% mu: estimated mean of each variable in input data                     %                                                                      
% totalVarPCs_M: total variance covered by the PCs_M user-defined       %
% principal components                                                  %
% reconstructions: reconstruction of the compressed input data to       %
% the user-defined 'M' principal components                             %
% totalMSE: total mean squared error between the original input data and%
% the reconstructed data after the compression to PCs_M user-defined    %
% principal components                                                  %
% varargout: optional output argument for the total root mean squared   %
% error if requested by the user                                        %
%                                                                       %
% It is possible to skip any of the output arguments by putting the     %
% tilda symbol (~), on these ones that you do not wish to be returned   %
% as outputs.                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [coeff, score, latent, mu, totalVarPCs_M, reconstructions, totalMSE, varargout] = PCAonData(data, PCs_M, typeOfData, varargin)
    
len = length(varargin);

if (nargin < 3)
    
    error('Invalid number of inputs');
    
elseif (len > 1)
    
    error('Invalid number of optional input arguments');
    
elseif ((len == 1) && (strcmpi(varargin{1}, 'rmse') ~= 1))
    
    error('Invalid type of optional input arguments');

else
    
    if ((strcmpi(typeOfData, 'im') ~= 1) && (strcmpi(typeOfData, 'ge') ~= 1))

        error('Invalid type of input data');
        
    end
    
    % number of observations
    observations = size(data, 2);

    [coeff, score, latent, ~, explained, mu] = pca(data');     

    % outputs adjusted only to PCs_M user-defined principal components
    coeff = coeff(:, (1 : PCs_M));
    score = score(:, (1 : PCs_M));
    latent = latent(1 : PCs_M);

    
    % total variance covered by the user-defined 'PCs_M' principle components
    totalVarPCs_M = sum(explained((1 : PCs_M), 1));

    %% Reconstruction of compressed input data

    mu = repmat(mu', 1, observations);
    reconstructions = coeff*score' + mu;

    %% Total Mean Squared Error/Root Mean Squared Error on the whole input dataset
    
    if (len == 1)  % Total Root Mean Squared Error
    
        [totalMSE, totalRMSE] = estimateTotalMSE(data, reconstructions, typeOfData, varargin); 
        varargout{:} = totalRMSE;
        
    else
        
        totalMSE = estimateTotalMSE(data, reconstructions, typeOfData); 
        
    end
        
end

end

