          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          %% Chantzi Efthymia - Deep Learning - Exercise 6 %%
          %%                     Task  A                   %%
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is used for creating the input and target datasets for  %
% the prediction modeling. The input dataset contain the feature vectors%
% t_(n+1), while the the target dataset contain the pairs (t_(n-1),t_n).%
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% images: 93 frames of a time-lapse microscopy movie for which the      %
% prediction modeling is performed                                      %
% movieAxis: index of the first frame from the time-lapse microscopy    %  
% movie on which the prediction model is built                          %
% stepMovieAxis: step indicating the size of the frames per movie       %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% inputs: feature vectors t_(n+1)                                       %
% targets: feature vector pairs (t_(n-1),t_n)                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [inputs, targets] = predictionDataset(images, movieAxis, stepMovieAxis)

len = length(movieAxis);

stepNewAxis = stepMovieAxis - 1;
newAxis = [1 : stepNewAxis : stepNewAxis*len];

for i = 1 : len

    inputs(:, newAxis(i) : newAxis(i) + (stepNewAxis - 1)) = images(:, movieAxis(i) : movieAxis(i) + (stepNewAxis - 1));
    targets(:, newAxis(i) : newAxis(i) + (stepNewAxis - 1)) = images(:, (movieAxis(i) + 1) : (movieAxis(i) + stepNewAxis));
    
end


end