       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %% Chantzi Efthymia - Deep Learning - Exercises 3,4,5,6 %%
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is used for plotting a user-defined number of original  %
% and reconstructed images in the same graphical window for visual      %
% inspection.                                                           %
% The user can select any of the available images by passing the index  % 
% of the first and last one. Additionally, the total mean square error  %
% between the whole original and reconstructed dataset is displayed on  %
% on the title of the figure.                                           %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% originalIm: original dataset of images                                %
% reconstructedIm: reconstructed dataset of images                      %
% indexVector: row vector with first and last index of the desired      %    
% subset of images for plotting, as elements                            %
% architectureVector: row vector with the structure of the deep network %
% as elements                                                           %
% reshapeVector: row vector with two elements; rows and columns of the  %
% images, respectively                                                  %
% totalMSE: total mean square error between the original & reconstructed%
% datasets                                                              %
% varargin: optional user-defined argument for cross validation. This   %
% optional parameter updates the version of this function from Exercise %
% 3. If this parameter is set to 'cv', then the default displaying of   %
% resulting figure is deactivated for the iterations of the double      %
% cross-validation procedure. Instead of this, in the respective .m file%
% for the execution of the double cross-validation, the user can invoke %
% the saving of the figure as a .png file. For more info see the file   %
% 'doubleCV.m'.                                                         %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% figure with the user-defined subset of original vs. reconstructed     %
% subset of images                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function  plotComparison(originalIm, reconstructedIm, indexVector, architectureVector, reshapeVector, totalMSE, varargin)

if (nargin < 5)
   
    error('Invalid number of inputs');
        
elseif (length(varargin) > 1)
    
    error('Invalid number of optional input arguments');
    
elseif ((length(varargin) == 1) && (strcmpi(varargin{1}, 'cv') ~= 1))
    
    error('Invalid type of optional input arguments');
    
else
    
    if (length(varargin) == 1)

        figure('Visible', 'off');

    else

        figure();

    end

    vec = [indexVector(1) : 1 : indexVector(end)];
    len = length(vec);

    for i = 1 : len

        subplot(2, len, i);
        imshow(reshape(originalIm(:, vec(i)), reshapeVector(1), reshapeVector(2)));

        subplot(2, len, (len + i));
        imshow(reshape(reconstructedIm(:, vec(i)), reshapeVector(1), reshapeVector(2)));

    end
    axes;
    if (isempty(architectureVector))

        h = title(sprintf('PCA Reconstruction\n MSE_{total} = %.5f', totalMSE));

    else

        g = sprintf('%d-', architectureVector);
        h = title(sprintf('Reconstruction by Deep network %s\n MSE_{total} = %.5f', g(1:end-1), totalMSE));

    end
    set(gca, 'Visible', 'off');
    set(h, 'Visible', 'on');
        
end

end