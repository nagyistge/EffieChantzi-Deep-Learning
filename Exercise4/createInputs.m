        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercises 2,3,4  %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates a subset of images if need be. Otherwise, it    % 
% keeps the initial set unchanged. It can be used for creating a        %
% smaller set of training and test images.                              %
% Additionally, this function assumes that initial set of images is in  %                                                                  
% the form of a cell array. For convenience, it turns the images into   %
% vectors and puts them in a matrix.                                    %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% setOfImages: cell array of (initial) set of images                    %
% usedImages: two-element row vector containing the indexes of the      %
% first and last image used in the subset.                              %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% newMatrixOfImages:  matrix that contains all the images as column     %
% vectors. The number of rows equals to the number of pixels, while     %
% the number of columns to the number of images in the set.             %
%                                                                       %   
%                                                                       %
% This function is used in all tasks of this exercise, wherever the     %
% input data must be in the form of a matrix.                           %
%                                                                       %
% IMPORTANT!!                                                           %
% The input argument vector "usedImages" must be the same as the input  % 
% argument vector "usedLabels" from function "createNewLabels", so      %
% that there is correct correspondence between images and their         %
% labels.                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function newMatrixOfImages = createInputs(setOfImages, usedImages)

indexes = [usedImages(1) : 1 : usedImages(2)];
subsetSize = length(indexes);

newMatrixOfImages = zeros(numel(setOfImages{1}), subsetSize);

for i = 1 : subsetSize
    
    newMatrixOfImages(:, i) = setOfImages{indexes(i)}(:);
    
end

end