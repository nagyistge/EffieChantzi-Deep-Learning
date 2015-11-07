%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Chantzi Efthymia - Deep Learning - Exercise 1  %%
%%                  Task D                        %%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function trains a deep network of stacked autoencoders on a given%
% training dataset.                                                     %
%                                                                       %
% Network Architecture:                                                 %
% {(size_input)-(size_HL1)-(size_HL2)-(size_HL3)-(size_output)}         %                                                     
%  The Autoencoder class requires Matlab R2015b.                        %                                                                      
%                                                                       %
% %%%% Inputs %%%%                                                      %
%                                                                       %
% trainingData: training dataset                                        %
% targetData: target dataset of outputs. In the case of autoencoders    %
% targetData = trainingData.                                            %
% vectorOfNeuronsHL: row vector of the 3 hidden layers' sizes           %                                                                      
%                                                                       %
% %%%% Outputs %%%%                                                     %
%                                                                       %
% reconstructedData: matrix of reconstructed input                      %
%                                                                       %
% After each autoencoder is trained, a plot with its achieved           %
% reconstruction in all dimensions is produced. There is a pause between%
% the consecutive autoencoders, so that you can see the results. Press  %
% any key to continue the execution. At the end, there is a plot of the %
% initial input vs. reconstructed data of the whole deep network of SAE %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function  reconstructedData = autoencoderHL3(trainingData, targetData, vectorOfNeuronsHL)

%% 1st Autoencoder

% train 1st autoencoder with input/training data
autoenc1 = trainAutoencoder(trainingData, vectorOfNeuronsHL(1), ...
'EncoderTransferFunction', 'satlin', ...
'DecoderTransferFunction', 'purelin');

% reconstructed data using 1st autoencoder
reconstructed = predict(autoenc1, trainingData);

% plot reconstruction of 1st autoencoder in all dimensions
figure();
for i = 1 : size(reconstructed, 1)
    
    subplot(size(reconstructed, 1), 1, i);
    plot(trainingData(i, :), 'm*');
    hold on ;
    plot(reconstructed(i, :), 'go');
    hold off;
    
end
legend('training data','reconstructed data','Location','Best');
axes;
h = title(sprintf('Input vs. Reconstruction(%d dimensions - Layer 1)', size(reconstructed, 1)));
set(gca, 'Visible', 'off');
set(h, 'Visible', 'on');

% encoded features of 1st autoencoder
features1 = encode(autoenc1, trainingData);
pause;

%% 2nd Autoencoder

% train 2nd autoencoder with the encoded features from 1st autoencoder
autoenc2 = trainAutoencoder(features1, vectorOfNeuronsHL(2), ...
'EncoderTransferFunction', 'satlin', ...
'DecoderTransferFunction','purelin', ...
'ScaleData', false);

% reconstructed data using 2nd autoencoder
reconstructed = predict(autoenc2, features1);

% plot reconstruction of 2nd autoencoder in all dimensions
figure();
for i = 1 : size(reconstructed, 1)
    
    subplot(size(reconstructed, 1), 1, i);
    plot(features1(i, :), 'm*');
    hold on ;
    plot(reconstructed(i, :), 'go');
    hold off;
    
end
legend('training data','reconstructed data','Location','Best');
axes;
h = title(sprintf('Input vs. Reconstruction(%d dimensions - Layer 2)', size(reconstructed, 1)));
set(gca, 'Visible', 'off');
set(h, 'Visible', 'on');

% encoded features of 2nd autoencoder
features2 = encode(autoenc2, features1);
pause;

%% 3rd Autoencoder

% train 3rd autoencoder with the encoded features from 2nd autoencoder
autoenc3 = trainAutoencoder(features2, vectorOfNeuronsHL(3), ...
'EncoderTransferFunction', 'satlin', ...
'DecoderTransferFunction','purelin', ...
'ScaleData', false);

% reconstructed data using 3rd autoencoder
reconstructed = predict(autoenc3, features2);

% plot reconstruction of 3rd autoencoder in all dimensions
figure();
for i = 1 : size(reconstructed, 1)
    
    subplot(size(reconstructed, 1), 1, i);
    plot(features2(i, :), 'm*');
    hold on ;
    plot(reconstructed(i, :), 'go');
    hold off;
    
end
legend('training data','reconstructed data','Location','Best');
axes;
h = title(sprintf('Input vs. Reconstruction(%d dimensions - Layer 3)', size(reconstructed, 1)));
set(gca, 'Visible', 'off');
set(h, 'Visible', 'on');

% encoded features of 3rd autoencoder
features3 = encode(autoenc3, features2);
pause;

%% 4th Autoencoder

% train a 4th (final) autoencoder of the same size as the input, so the
% reconstructed data have the same size as the initial data. This last
% autoencoder is trained with the encoded features from  3rd autoencoder
autoenc4 = trainAutoencoder(features3, size(trainingData, 1), ...
'EncoderTransferFunction', 'satlin', ...
'DecoderTransferFunction','purelin', ...
'ScaleData', false);

% reconstructed data using 4th autoencoder
reconstructed = predict(autoenc4, features3);

% plot reconstruction of 4th autoencoder in all dimensions
figure();
for i = 1 : size(reconstructed, 1)
    
    subplot(size(reconstructed, 1), 1, i);
    plot(features3(i, :), 'm*');
    hold on ;
    plot(reconstructed(i, :), 'go');
    hold off;
    
end
legend('training data','reconstructed data','Location','Best');
axes;
h = title(sprintf('Input vs. Reconstruction(%d dimensions - Output Layer)', size(reconstructed, 1)));
set(gca, 'Visible', 'off');
set(h, 'Visible', 'on');


%% Deep Network of SAE (Stacked Autoencoders)

% form deepnet network by stacking all traied autoencoders and perform 
% supervised learning
stackNet = stack(autoenc1, autoenc2, autoenc3, autoenc4);
view(stackNet);

% train deepnet network
deepnet = train(stackNet, trainingData, targetData);

% test deepnet network on the initial training data
reconstructedData = deepnet(trainingData);

% mean squared reconstruction error
mseError = mse(trainingData - reconstructedData);

% plot input vs. reconstructed data of the trained deepnet network of SAE 
figure();
plot3(trainingData(1, :), trainingData(2, :), trainingData(3, :), 'm*', 'MarkerSize', 10);
hold on;
plot3(reconstructedData(1, :), reconstructedData(2, :), reconstructedData(3, :), 'go', 'MarkerSize', 10);
grid on;
xlabel('b_1_-', 'fontweight', 'bold');
ylabel('b_2_-', 'fontweight', 'bold');
zlabel('b_3_-', 'fontweight', 'bold');
title(sprintf('Autoencoder Neural Network (%d-%d-%d-%d-%d)\n Recostruction MSE = %f%', size(trainingData, 1), vectorOfNeuronsHL(1), vectorOfNeuronsHL(2), vectorOfNeuronsHL(3), size(trainingData, 1), mseError));
legend({'Training Data', 'Reconstructed Data'}, 'Location', 'northeast', 'FontSize', 8, 'FontWeight', 'bold');
hold off;

end