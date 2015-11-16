        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise 2  %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function plots the images in the output of the encoder of the    %
% 1st autoencoder in a deep neural network of stacked autoencoders(SAE).%
% As mentioned, this requires that the deep neural network is trained   %
% on image data.                                                        %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% weights: produced weights of the 1st hidden layer as a                %
% (size_image)-by-(size_hiddenLayer_1) matrix (i.e., 784x25)            %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% common figure of all images in the output of the first hidden layer   %
% of the stacked autoencoders.                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [] = plotWeightsAsImages(weights)

% transpose of the gained weights in the 1st hidden layer
WeightsAsImage = weights';

% regain the dimensions of images: rows and columns
dim = sqrt(size(WeightsAsImage, 1));

% division of the figure, so that all images of the 1st hidden layer are
% displayed in one common plot. The subplot has 5 rows and k columns, which
% are determined automatically.
k = ceil(size(WeightsAsImage, 2)/5);
figure();
for i = 1 : size(WeightsAsImage, 2)
    
    subplot(5, k, i);
    imshow(reshape(WeightsAsImage(:, i), dim, dim));
    
end

 axes;
 h = title('Weights as Images in 1^{st} hidden layer of Stacked Autoencoders');
 set(gca, 'Visible', 'off');
 set(h, 'Visible', 'on');
end