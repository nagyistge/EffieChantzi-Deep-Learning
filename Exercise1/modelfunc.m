%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Chantzi Efthymia - Deep Learning - Exercise 1  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function defines the two requested model functions:              %      
% x = b1*t1 + b2*(t1^2) and x = b1*t1 + b2*t2                           %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% b: basis matrix                                                       %
% t: matrix of score values                                             %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% data: generated data  in the form of column vectors, where each one   %
% responds to a sample. Each sample, has as many rows as the basis      %
% matrix b.                                                             %    
%                                                                       %
% The provided function recognizes the dimensions of the given b and t  %
% and generates the appropriate data. The generation of matrices b and  %
% t is handled by the function 'generateDataset' in the file            %
% 'generateDataset.m'                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function data = modelfunc(b, t)

rb = size(b, 1);
[rt, ct] = size(t);
data = zeros(rb, ct);

if (rt == 1) 
    
    for i = 1 : ct
       
       data(:, i) = b(:, 1)*t(i) + b(:, 2)*(t(i)^2); % x = b1*t1 + b2*(t1^2)
        
    end
    
elseif (rt == 2)
    
    for i = 1 : ct
        
        data(:, i) = b(:, 1)*t(1, i) + b(:, 2)*t(2, i); % x = b1*t1 + b2*t2
        
    end
    
else
    
    error('Invalid dimensions'); % in terms of the two requested models
    
end

end