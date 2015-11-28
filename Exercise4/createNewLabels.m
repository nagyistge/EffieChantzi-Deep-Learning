        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercises 2,3,4  %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates the new labels for the images, as belonging to  %
% class "3" or "other". The provided labels are stored in a 10-by-5000  %
% matrix. The newly created labels are stored in a 1-by-(size_of_set)   %
% matrix, where "size_of_set" can be either a subset or the whole set   %
% of images. Depending on the structure of the initital label matrix,   %
% if its third row for a specific column(=image) has the element 1,     %
% then the image depicts digit 3, otherwise an other digit.             %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% labels_10digits: 10-by-5000 matrix of labels for all ten digits (0-9) %
% usedLabels: two-element of column vector containg the indexes         % 
% of the first and last image used in the subset.                       %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% labels:  1-by-(size_of_set) matrix containg the labels for "3"(1)     %
% and "other"(0) images.                                                %
%                                                                       %
%                                                                       %
% IMPORTANT!!                                                           %
% The input argument vector "usedLabels" must be the same as the input  % 
% argument vector "usedImages" from function "createInputs", so that    %
% that there is correct correspondence between images and their         %
% respective labels.                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function labels = createNewLabels(labels_10digits, usedLabels)

% size of subset
subsetSize = (usedLabels(2) - usedLabels(1)) + 1;

% extract the third row of the used images, which corresponds to digit "3"
sample = labels_10digits(3, (usedLabels(1) : usedLabels(2)));

labels = zeros(1, subsetSize);
for i = 1 : subsetSize
    
    if (sample(1, i) == 1) % digit "3"
        
        labels(1, i) = 1;  % output class "1"
        
    end
    
end

end