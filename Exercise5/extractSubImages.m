           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %% Chantzi Efthymia - Deep Learning - Exercise 5  %%
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function extracts a subimage/patch of all the extracted frames   %
% from the time-lapse movie dataset that is used for Exercise 5.        %
% This equally-sized extracted patches from all used frames of the      %
% respective video objects are stacked together and form the final      %
% d-dimensional image dataset.                                          %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% images: extracted images/frames from the video objects                %
% step_height: positive integer number indicating the length of the     % 
% extracted patch in the direction of rows                              %
% step_width: positive integer number indicating the length of the      % 
% extracted patch in the direction of columns                           %
% upperLeftRow: positive integer number indicating the row coodinate of %
% the upper left corner of the extracted patch                          %
% upperLeftCol: positive integer number indicating the column coodinate %
% of the upper left corner of the extracted patch                       %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% subimages: extracted subimages of the respective input images         %
%                                                                       %
%                                                                       %
% IMPORTANT !!!                                                         %
% This function is nested in the function 'datasetFromAllMovies.m',     %
% where the coordinates of the upper left corner of the extracted patch %
% are dynamically decided. Moreover, the input images are 4-D, where the%
% 3rd dimension respond to the 3 RGB channels. However, the provided    %
% movies and thus, images are grayscale. For this reason and for        %
% reducing the computational complexity, only one color channel is kept %
% for the needs of Exercise 5. Finally, the extracted subimages are     %
% converted from unint8 to double with intensity range [0, 1].          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function subimages = extractSubImages(images, step_height, step_width, upperLeftRow, upperLeftCol)

% 4th dimension includes the number of frames
frames = size(images, 4);

% initialize the matrix that stores the extracted subimages
subimages = zeros(step_height, step_width, frames);

% extract and convert subimages for each one of the input images/frames
for i = 1 : frames
   
    subimages(:, :, i) = double(images((upperLeftRow : (upperLeftRow + (step_height - 1))), (upperLeftCol : (upperLeftCol + (step_width - 1))), 1, i))/255;
    
end

% reshape the extracted patches in the appropriate d-dimensional input
% vectors
subimages = reshape(subimages, step_height*step_width, frames);

end