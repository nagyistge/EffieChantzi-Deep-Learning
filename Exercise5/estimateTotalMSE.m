        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercises 3,4,5 %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function estimates the total mean squared error either between   %
% two image datasets or between two gene expression data matrices.      %
% In the case of the image datasets, the total mean squared error is    % 
% estimated, while for the gene expression datasets the average         %
% relative error is assessed, as requested by Exercise 5.               %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% inputs: reference/original dataset of images                          %
% reconstructions: dataset of images for comparison with the reference  %
% dataset                                                               %
% typeOfData: string that defines the type of input data matrix; 'im'   %
% for images and 'ge' for gene expression data. This input argument is  %
% necessary, since the formula for the estimation of the mean squared   %
% error differs between images and gene expression data.                %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% totalMSE: total mean square error between reference and altered       %
% datasets                                                              %
%                                                                       %   
%                                                                       %
% This function is nested in function "PCAonImages". However, it is     %
% also used separately to estimate the total mean square error between  %
% inputs and outputs(reconstructed inputs) of employed deep networks.   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function totalMSE = estimateTotalMSE(inputs, reconstructions, typeOfData)

obs = size(inputs, 2);
MSE = zeros(1, obs);

if(strcmpi(typeOfData, 'im')) % images
   
    for i = 1 : obs
   
        MSE(1, i) = mean((reconstructions(:, i) - inputs(:, i)).^2); 
    
    end
    
elseif(strcmpi(typeOfData, 'ge')) % gene expression data
    
    for i = 1 : obs
   
        MSE(1, i) = norm(inputs(:, i) - reconstructions(:, i), 1)/norm(inputs(:, i), 1);
    
    end
    
else
    
    error('Invalid type of data');
    
end

totalMSE = mean(MSE);

end