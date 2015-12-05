           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %% Chantzi Efthymia - Deep Learning - Exercise 5  %%
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates the image dataset extracted from the time-lapse %
% microscopy movies, which is required for Exercise 5. More precisely,  %
% it creates a d-dimensional matrix of n image vectors, where d is equal%
% to the total number of pixels of the extracted patches and n is the   %
% number of the total frames/images used.                               %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% movieObjArray: object array of movie objects returned by the function %
% 'createMovieObjects'                                                  %
% useFrames: number of total extracted frames from all video objects    %
% step_height: positive integer number indicating the length of the     % 
% extracted patch in the direction of rows                              %
% step_width: positive integer number indicating the length of the      % 
% extracted patch in the direction of columns                           %
% offset: positive integer number indicating the movement of the middle %
% point(centre) in the image leftwise. In other words, the middle row   %
% and column of each image minus this number respond to the upper left  % 
% corner of the extracted patch.                                        %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% inputs: d-dimensional matrix of n input vectors; image dataset        %
% movieAxis: axis of frames indices used for the final merge of the     %
% extracted patches from all movie objects to the final image dataset   %
% 'inputs'. This is returned by the function, since it can be useful    %
% for visual inspection of the extracted frames according to the video  %
% that they belong to.                                                  %
%                                                                       %
%                                                                       %
% IMPORTANT !!!                                                         %
% This function is used for the creation of the training and test set   %
% by using a different number for the 'offset' argument, so that the    %
% the resulting patches for the two sets contain different features.    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [inputs, movieAxis] = datasetFromAllMovies(movieObjArray, usedFrames, step_height, step_width, offset)

% number of movie objects
numOfMovies = length(movieObjArray);

% total number of image vectors in the dataset
totalImages = (usedFrames*numOfMovies);

% initialize dataset matrix
inputs = zeros((step_height*step_width), totalImages);

% axis with indices for each video to be used for the creation of the final
% matrix of vectors of inputs
movieAxis = [1 : usedFrames : totalImages];

for i = 1 : numOfMovies
   
    images = read(movieObjArray(1, i).Value, [1 usedFrames]);
    subimages = extractSubImages(images, step_height, step_width, (floor(size(images, 1)/2) - offset), (floor(size(images, 2)/2) - offset));
    inputs(:, movieAxis(i) : movieAxis(i) + (usedFrames - 1)) = subimages;
    
end


end
