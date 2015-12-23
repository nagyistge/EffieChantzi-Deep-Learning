          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          %% Chantzi Efthymia - Deep Learning - Exercise 6 %%
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function estimates the total mean squared error and total root   %
% mean squared error(if requested), either between two (image) datasets %
% or between two gene expression datasets.                              %
% In the case of the image/other datasets, the total mean squared error %
% is estimated, while for the gene expression datasets the average      %
% relative error is calculated.                                         %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% inputs: reference/original dataset                                    %
% reconstructions: reconstructed dataset                                %                                                      
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
% totalMSE: total mean square error between reference and reconstructed %
% datasets                                                              %
% varargout: optional output argument for the total root mean squared   %
% error if requested by the user                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [totalMSE, varargout] = estimateTotalMSE(inputs, reconstructions, typeOfData, varargin)


if (nargin < 3)
   
    error('Invalid number of input arguments\n');
    
elseif (length(varargin) > 1)
    
    error('Invalid number of optional input arguments');
    
elseif ((length(varargin) == 1) && (strcmpi(varargin{1}, 'rmse') ~= 1))
    
    error('Invalid type of optional input arguments');
    
   
else
    
    len = length(varargin);
    obs = size(inputs, 2);
    MSE = zeros(1, obs);
    
    if (len == 1)
       
        RMSE = zeros(1, obs); 
        
    end

    if(strcmpi(typeOfData, 'im')) % images/other

        for i = 1 : obs

            MSE(1, i) = mean((inputs(:, i) - reconstructions(:, i)).^2); 
            
            if (len == 1)
            
                RMSE(1, i) = sqrt(MSE(1, i)); % rmse
                
            end

        end

    elseif(strcmpi(typeOfData, 'ge')) % gene expression data

        for i = 1 : obs

            MSE(1, i) = norm(inputs(:, i) - reconstructions(:, i), 1)/norm(inputs(:, i), 1);

        end

    else

        error('Invalid type of data');

    end

    totalMSE = mean(MSE);
    
    if(len == 1)
       
        totalRMSE = mean(RMSE);
        varargout{:} = totalRMSE;
        
    end
    
    
end


end