        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise 3  %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function estimates the total mean square error between two image %
% datasets, which are defined as two matrices, where rows correspond to %
% dimensions(pixels) and columns to images(observations).               %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% inputs: reference/original dataset of images                          %
% reconstructions: dataset of images for comparison with the reference  %
% dataset                                                               %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% totalMSE: total mean square error between reference and altered       %
% datasets of images                                                    %
%                                                                       %   
%                                                                       %
% This function is nested in function "PCAonImages". However, it is     %
% also used separately to estimate the total mean square error between  %
% inputs and outputs(reconstructed inputs) of employed deep networks.   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function totalMSE = estimateTotalMSE(inputs, reconstructions)

obs = size(inputs, 2);
MSE = zeros(1, obs);

for i = 1 : obs
   
    MSE(1, i) = immse(reconstructions(:, i), inputs(:, i));      %MSE(1, i) = mean((reconstructions(:, i) - inputs(:, i)).^2); or MSE(1, i) = (sum((reconstructions(:, i) - inputs(:, i)).^2))/(size(inputs(:, i), 1))
    
end


totalMSE = mean(MSE);
fprintf('Total MSE of Reconstruction: %.5f%\n', totalMSE);
fprintf('\n');

end