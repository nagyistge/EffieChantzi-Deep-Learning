        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercises 4,5 %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is used for creating corrupted images by adding pepper  %
% noise(setting some randomly selected pixels to zero), which are used  %
% only during the pre-training phase of the denoising stacked           %
% autoncoders. It is used in each input layer of all the autoencoders   %
% that constitute the deep neural network.                              %
%                                                                       %
%                                                                       %        
% %%%% Inputs %%%%                                                      %
% images: original/uncorrupted dataset of images                        %
% d: noise density, a number greater than 0 and smaller or equal to 1   %  
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% images: corrupted images with pepper noise of density d               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function images = pepperNoise(images, d)

if (nargin < 2)
   
    error('Invalid number of input arguments');
    
else
        
   if (d > 0) && (d <= 1)
        
        
        [dim, obs] = size(images);

        % number of corrupted pixels based on the user-defined noise density 'd'
        numOfCorruptedPixels = round(d*dim);

        for i = 1 : obs

            % generation of random positions for the disabled pixels for each image of
            % the dataset
            r = randi([1, dim], 1, numOfCorruptedPixels);

            for j = 1 : numOfCorruptedPixels

                images(r(j), i) = 0;

            end

        end
       
        
    else
        
        error('Invalid range of noise density.\n It should be greater than 0 and smaller or equal to 1.');
        
    end
    
end

end