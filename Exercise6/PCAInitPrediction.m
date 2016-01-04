         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         %% Chantzi Efthymia - Deep Learning - Exercise 6 %%
         %%                    Task A                     %%
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function initializes the weights and biases of a deep neural     %
% network with 1 or 2 hidden layers, meaning 3 and 4 layers in total,   %
% based on principal component analysis (PCA).                          %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% PCAvector: row vector with the sizes of hidden(s) and output layers   %
% s: user-defined parameter for the diagonal matrix D                   %
% data: multi-dimensional input data, where rows respond to dimensions  %
% and columns respond to observations                                   %
% activationFunction: activation function of the neurons in hidden      %
% layers. Two options as an inserted string: 'logsig' or 'tansig'.      %
% typeOfData: string that defines the type of input data matrix; 'im'   %
% for images/other and 'ge' for gene expression data. This input        %
% argument is necessary, since the formula for the estimation of the    %
% mean squared error differs for gene expression data.                  %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% WeightsBiases: a 1-by-m cell array, where m is the number of          %
% hidden plus output layers of the network multiplied by 2. The first   %
% m/2 cells contain the weights in increasing order, while the second   %
% m/2 cells contain the biases in increasing order.                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function WeightsBiases = PCAInitPrediction(PCAvector, s, data, activationFunction, typeOfData)

len = length(PCAvector);

if (len == 2)                   % 1 hidden layer (3 layers in total)
    
    WeightsBiases = cell(1, 4);
    
   %% 1st hidden layer
    [coeff, ~, latent, mu, ~, ~, ~] = PCAonData(data, PCAvector(1), typeOfData);

    % weights and biases for 1st HL
    B1 = coeff(:, (1 : PCAvector(1))); 
    c = size(B1, 2);

    sqrt_lamda = sqrt(latent);
    factor = (ones(c, 1)*s)./sqrt_lamda;
    D1 = diag(factor);

    W1 = D1*B1';
    b1 = -W1*mu;

    a_n = W1*data + b1;

    if (strcmpi(activationFunction, 'logsig') == 1)

        z_n = logsig(a_n);

    elseif (strcmpi(activationFunction, 'tansig') == 1)

        z_n = tansig(a_n);

    else

        error('Invalid type of activation function. Choose between "logsig" and "tansig".\n');

    end
    
    %% Output Layer
    
    [coeff2, ~, latent2, mu2, ~, ~, ~] = PCAonData(z_n, PCAvector(2), typeOfData);

    % weights and biases for 2nd HL
    B2 = coeff2(:, (1 : PCAvector(2))); 
    c2 = size(B2, 2);

    sqrt_lamda2 = sqrt(latent2);
    factor2 = (ones(c2, 1)*s)./sqrt_lamda2;
    D2 = diag(factor2);

    W2 = D2*B2';
    b2 = -W2*mu2;


    %% Assign all weight and biases matrices to cell structure as output
    WeightsBiases{1, 1} = W1;
    WeightsBiases{1, 2} = W2;
    WeightsBiases{1, 3} = b1;
    WeightsBiases{1, 4} = b2;
    
    
elseif (len == 3)               % 2 hidden layers (4 layers in total)
    
    WeightsBiases = cell(1, 6);
    
    %% 1st hidden layer
    
    [coeff, ~, latent, mu, ~, ~, ~] = PCAonData(data, PCAvector(1), typeOfData);

    % weights and biases for 1st HL
    B1 = coeff(:, (1 : PCAvector(1))); 
    c = size(B1, 2);

    sqrt_lamda = sqrt(latent);
    factor = (ones(c, 1)*s)./sqrt_lamda;
    D1 = diag(factor);

    W1 = D1*B1';
    b1 = -W1*mu;

    a_n = W1*data + b1;

    if (strcmpi(activationFunction, 'logsig') == 1)

        z_n = logsig(a_n);

    elseif (strcmpi(activationFunction, 'tansig') == 1)

        z_n = tansig(a_n);

    else

        error('Invalid type of activation function. Choose between "logsig" and "tansig".\n');

    end
    
    %% 2nd Hidden Layer

    [coeff2, ~, latent2, mu2, ~, ~, ~] = PCAonData(z_n, PCAvector(2), typeOfData);

    % weights and biases for 2nd HL
    B2 = coeff2(:, (1 : PCAvector(2))); 
    c2 = size(B2, 2);

    sqrt_lamda2 = sqrt(latent2);
    factor2 = (ones(c2, 1)*s)./sqrt_lamda2;
    D2 = diag(factor2);

    W2 = D2*B2';
    b2 = -W2*mu2;

    a_n2 = W2*z_n + b2;

    if (strcmpi(activationFunction, 'logsig') == 1)

        z_n2 = logsig(a_n2);

    elseif (strcmpi(activationFunction, 'tansig') == 1)

        z_n2 = tansig(a_n2);

    else

        error('Invalid type of activation function. Choose between "logsig" and "tansig".\n');

    end

    %% Output Layer
    
    [coeff3, ~, latent3, mu3, ~, ~, ~] = PCAonData(z_n2, PCAvector(3), typeOfData);

    % weights and biases for 2nd HL
    B3 = coeff3(:, (1 : PCAvector(3))); 
    c3 = size(B3, 2);

    sqrt_lamda3 = sqrt(latent3);
    factor3 = (ones(c3, 1)*s)./sqrt_lamda3;
    D3 = diag(factor3);

    W3 = D3*B3';
    b3 = -W3*mu3;
    
    %% Assign all weight and biases matrices to cell structure as output
    WeightsBiases{1, 1} = W1;
    WeightsBiases{1, 2} = W2;
    WeightsBiases{1, 3} = W3;
    WeightsBiases{1, 4} = b1;
    WeightsBiases{1, 5} = b2;
    WeightsBiases{1, 6} = b3;
    
else
    
    error('Invalid number of hidden layers\n');
    
end


end