          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          %% Chantzi Efthymia - Deep Learning - Exercise 6 %%
          %%                     Task  B                   %%
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is used for creating random weights and biases drawn    % 
% from the interval [-alpha, alpha], for a neural network with one or   %
% more hidden layers.                                                   %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% alpha: user-defined parameter declaring the range of the interval     %
% architecture: row vector with the network structure as elements       %  
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% WeightsBiases: a 1-by-m cell array, where m is the number of          %
% hidden plus output layers of the network multiplied by 2. The first   %
% m/2 cells contain the weights in increasing order, while the second   %
% m/2 cells contain the biases in increasing order.                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function WeightsBiases = randomIntervalInit(alpha, architecture)

interval = [-alpha alpha];
len = length(architecture);
WeightsBiases = cell(1, 2*(len - 1));

for i = 1 : (2*(len - 1))
   
   if (i < len)        % weights of layers
       
       WeightsBiases{1, i} = (interval(2) - interval(1)).*rand(architecture(i + 1), architecture(i)) + interval(1);
       
   else                 % biases of layers
       
       WeightsBiases{1, i} = (interval(2) - interval(1)).*rand(architecture(i - 1), 1) + interval(1);
       
   end
    
    
end

end