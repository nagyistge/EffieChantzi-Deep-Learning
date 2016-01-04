        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise 3  %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
% totalMSE: total mean square error between the original & reconstructed%
% datasets                                                              %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% figure with the user-defined subset of original vs. reconstructed     %
% subset of images                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function  plotComparison(originalIm, reconstructedIm, indexVector, architectureVector, totalMSE)

figure();
vec = [indexVector(1) : 1 : indexVector(end)];
len = length(vec);

for i = 1 : len
    
    subplot(2, len, i);
    imshow(reshape(originalIm(:, vec(i)), 28, 28));
            
    subplot(2, len, (len + i));
    imshow(reshape(reconstructedIm(:, vec(i)), 28, 28));
        
end
axes;
if (isempty(architectureVector))
   
    h = title(sprintf('PCA Reconstruction\n MSE_{total} = %.4f%', totalMSE));
    
else
   
    g = sprintf('%d-', architectureVector);
    h = title(sprintf('Reconstruction by Deep network %s\n MSE_{total} = %.4f%', g(1:end-1), totalMSE));
    
end
set(gca, 'Visible', 'off');
set(h, 'Visible', 'on');

end