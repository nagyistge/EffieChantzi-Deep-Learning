        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise 4 %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is used for creating the subset of images needed for    %
% the double cross-validation procedure of task B. The newly created    %
% subset contains all the "3" examples from the whole dataset, which in %
% the case of digittrain_dataset is 493. The total number of examples   %
% in the neewly created subset is a user-defined input argument.        %
% This means that the number of "other" examples in the subset is       %
% (size_of_user_defined_subsset - 493).                                 %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% images: initial dataset of images                                     %
% labels: vector of labels concerning the discrimination between "3"    %
% and "other" examples                                                  %
% obs_subset: number of examples for the newly created subset.          %    
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% images_subset: newly created subset of images as a                    %
% (dimensions-by-obs_subset) matrix of vectors. The dimensions are      %
% are inherited from the initial dataset.                               %
%                                                                       %
%                                                                       %
%                                                                       %
% IMPORTANT!!!                                                          %
% The input agument 'obs_subset' must be carefully selected according   %
% to the total structure of the performed double cross-validation.      %
% More precisely, it should be a number sufficiently larger than the    %
% dimensions(pixels), so that it guarantees that after the outer & inner%
% partitions, it is still greater than the dimensions. Otherwise, pca   %
% will fail, since the observations will be less than the dimensions,   % 
% resulting in a coefficient matrix which is not of full rank.          %
% In general, increase this number as you decrease the number of outer  %
% folds.                                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function images_subset = createSubsetForCrossValidation(images, labels, obs_subset)

dim = size(images, 1);

% indices of input images with digit "3" and "others"
[r_3, c_3] = find(labels == 1);
[r_others, c_others] = find(labels == 0);

images_subset = zeros(dim, obs_subset);
count = 0;
for i = 1 : obs_subset
    
    if(i <= 493) % 493 is the total number of "3" images in the digittrain_dataset
        
        images_subset(:, i) = images(:, c_3(i));
        
    else
        
        count = count + 1;
        images_subset(:, i) = images(:, c_others(count));
        
    end
    
end

end