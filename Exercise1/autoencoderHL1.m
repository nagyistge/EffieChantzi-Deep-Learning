%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Chantzi Efthymia - Deep Learning - Exercise 1  %%
%%                  Task D                        %%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function trains an autoencoder with one hidden layer on a given  %
% training dataset.                                                     %
%                                                                       %
% Network Architecture:                                                 %
% {(size_input)-(size_HL1)-(size_output)}                               %                                                     
%  The Autoencoder class requires Matlab R2015b.                        %                                                                      
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
%                                                                       %
% trainingData: training dataset                                        %
% numNeurons: number of neurons in the hidden layer                     %
%                                                                       %
% varargin: optional set of inputs defined by user as described in      % 
% http://se.mathworks.com/help/nnet/ref/trainautoencoder.html           %
% If varargin is not provided by the user, the default values of the    %
% Autoencoder class are set, as described in the aforementioned link.   %
%                                                                       %
% %%%% Outputs %%%%                                                     %
%                                                                       %
% reconstructedData: matrix of reconstructed input                      %
% Two figures are generated:                                            %
% i) achieved reconstruction in each one of all different dimensions    %
% ii) 3-D plot of training vs. reconstructed data and plot of features  %
% in the hidden layer                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function reconstructedData = autoencoderHL1(trainingData, numNeurons, varargin)

if(nargin < 2)
   
    error('Invalid number of inputs');
        
elseif(nargin == 2)
    
    autoenc = trainAutoencoder(trainingData, numNeurons); % train a sparse autoencoder with default settings
    
else 
    
    autoenc = trainAutoencoder(trainingData, numNeurons, varargin{:}); % further user-specified options
     
end

view(autoenc);

% reconstruct data using the trained autoencoder
reconstructedData = predict(autoenc, trainingData);

% get encoded data
features = encode(autoenc, trainingData);

% mean squared reconstruction error
mseError = mse(trainingData - reconstructedData);

% plot training vs. reconstructed data in each one of the different 
% dimensions to inspect discrepancies
figure();
for i = 1 : size(trainingData, 1)
subplot(size(trainingData, 1), 1, i);
plot(trainingData(i, :), 'm*');
hold on;
plot(reconstructedData(i, :), 'go');
hold off;
end
legend('training data', 'reconstructed data', 'Location', 'best');
axes;
h = title(sprintf('Input vs. Reconstruction in each one of %d dimensions', size(trainingData, 1)));
set(gca, 'Visible', 'off');
set(h, 'Visible', 'on');


% plot actual vs. reconstructed values
figure();
subplot(1, 2, 1);
plot3(trainingData(1, :), trainingData(2, :), trainingData(3, :), 'm*', 'MarkerSize', 10);
hold on;
plot3(reconstructedData(1, :), reconstructedData(2, :), reconstructedData(3, :), 'go', 'MarkerSize', 10);
grid on;
xlabel('b_1_-', 'fontweight', 'bold');
ylabel('b_2_-', 'fontweight', 'bold');
zlabel('b_3_-', 'fontweight', 'bold');
title(sprintf('Autoencoder Neural Network (3-%d-3)\n Recostruction MSE = %f%', numNeurons, mseError));
legend({'Training Data', 'Reconstructed Data'}, 'Location', 'northeast', 'FontSize', 8, 'FontWeight', 'bold');
hold off;

% plot weights/features in the hidden layer
subplot(1, 2, 2);
plot(features(1, :), features(2, :), 'go', 'Markersize', 10);
grid on;
xlabel('w_1', 'fontweight', 'bold');
ylabel('w_2', 'fontweight', 'bold');
title('Weights in the hidden layer', 'fontweight', 'bold');


end