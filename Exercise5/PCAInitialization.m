        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercises 3,4,5  %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function initializes the weights and biases of a user-defined    %
% 3-hidden layer deep network of multi-dimensional inputs and outputs.  %
% This practically means that four weights and four biases matrices     %
% are obtained as outputs, which can then used for initialization of a  %
% customized deep neural network. The initialization is based on PCA of %
% the input data and pertains to two different activation functions of  %
% the hidden layers; logistic sigmoid ('logsig') and hyperbolic tangent %
% sigmoid ('tansig').                                                   %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% PCAvector: row vector of the 3 hidden layer sizes                     %
% s: user-defined parameter for the diagonal matrix D                   %
% data: multi-dimensional input data, where rows respond to dimensions  %
% and columns respond to observations                                   %
% activationFunction: activation function of the neurons in hidden      %
% layers. Two options as an inserted string: 'logsig' or 'tansig'.      %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% W1: weights matrix for input layer (size_HL1xinitial_dimensions)      %
% W2: weights matrix for 1st hidden layer (size_HL2xsize_HL1)           %
% W3: weights matrix for 2nd hidden layer (size_HL3xsize_HL2)           %
% W4: weights matrix for output layer (initial_dimensionsxsize_HL3)     %
% b1: biases matrix for input layer (size_HL1xobservations)             %
% b2: biases matrix for 1st hidden layer (size_HL2xobservations)        %
% b3: biases matrix for 2nd layer (size_HL3xobservations)               %
% b4: biases matrix for output layer (initial_dimensionsxobservations)  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [W1, W2, W3, W4, b1, b2, b3, b4] = PCAInitialization(PCAvector, s, data, activationFunction, typeOfData)

% number of observations/samples of input data
samples = size(data, 2);


%% 1st hidden layer
[coeff, ~, latent, mu, ~, ~, ~] = PCAonData(data, PCAvector(1), typeOfData);

% weights and biases for 1st HL
B1_transpose = coeff(:, (1 : PCAvector(1)))'; 
c = size(B1_transpose, 2);

sqrt_lamda = sqrt(latent);
factor = (ones(c, 1)*s)./sqrt_lamda;
D1 = diag(factor);
W1 = B1_transpose*D1;
b1 = -W1*mu;

a_n = W1*(coeff'*(data - mu)) + b1;

if (strcmpi(activationFunction, 'logsig') == 1)
    
    z_n = logsig(a_n);
    mode = 1;
    
elseif (strcmpi(activationFunction, 'tansig') == 1)
    
    z_n = tansig(a_n);
    mode = 2;
    
else
   
    error('Invalid type of activation function. Choose between "logsig" and "tansig".\n');
    
end

%% 2nd Hidden Layer

[coeff2, ~, latent2, mu2, ~, ~, ~] = PCAonData(z_n, PCAvector(2), typeOfData);

% weights and biases for 2nd HL
B2_transpose = coeff2(:, (1 : PCAvector(2)))'; 
c2 = size(B2_transpose, 2);

sqrt_lamda2 = sqrt(latent2);
factor2 = (ones(c2, 1)*s)./sqrt_lamda2;
D2 = diag(factor2);

W2 = B2_transpose*D2;
b2 = -W2*mu2;

%% 3rd Hidden Layer

S = sparse(D2);
C = inv(S);
inverseD2 = full(C);
o3 = ones(PCAvector(2), samples);

if (mode == 1) % logsig
    
    W3 = 4*(W2*inverseD2)';
    b3 = mu2 - ((1/2)*W3)*o3; 
    
else % tansig
    
   W3 = (W2*inverseD2)';
   b3 = mu2 - W3*o3; 
    
end

%% Output Layer

S = sparse(D1);
C = inv(S);
inverseD1 = full(C);
o4 = ones(PCAvector(1), samples);

if (mode == 1) % logsig
   
    W4 = 4*(W1*inverseD1)';
    b4 = mu - ((1/2)*W4)*o4; 
    
else % tansig
    
    W4 = (W1*inverseD1)';
    b4 = mu - W4*o4;
    
end

end