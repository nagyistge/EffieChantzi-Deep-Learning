%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Chantzi Efthymia - Deep Learning - Exercise 1  %%
%%                  Task B, C                     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function generates the datasets in terms of a model function,    %
% plots the generated dataset in 3-D, and performs PCA on it.           %
%                                                                       %
% %%%% Inputs %%%%                                                      %
%                                                                       %
% numOfSamples: number of samples (columns of the generated dataset)    %
% dimBasisVector: row dimensions of the basis vectors                   %                 
% numOfBasisVectors: number of basis vectors (i.e., (b1, b2)-> 2)       %
% numOfRandomVar: number of random variables (score values)             %
% func: defines the requested model function (user-defined as string)   %
%                                                                       %
% varargin: optional set of inputs defined by user. Specifically,       %
% there are two options:                                                %
% i) 'plotDataset' -> plots the generated dataset in 3-D                %
% ii) 'pca' -> performs PCA and plot the dataset in 3-D and after PCA   %
%                                                                       %
% %%%% Outputs %%%%                                                     %
%                                                                       %
% dataset: generated dataset                                            %
% varargout: optional output containing the eigenvalues if the optional %
% input argument 'pca' is set                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [dataset, varargout] = generateDataset(numOfSamples, dimBasisVector, numOfBasisVectors, numOfRandomVar, func, varargin)

if ((nargin < 5) || (nargin > 6))
    
    error('Invalid number of inputs');
    
else
    
    % basis vectors
    b = randn(dimBasisVector, numOfBasisVectors);

    % random variables
    t = randn(numOfRandomVar, numOfSamples);

    % user-specified model function
    dataset = feval(func, b, t);
    
    if (length(varargin) == 1)
       
        if (strcmpi(varargin{1}, 'plotDataset') == 1)
           
            % plot generated dataset in 3-D
            figure();
            plot3(dataset(1, :), dataset(2, :), dataset(3, :), '*', 'color', 'm', 'MarkerSize', 10);
            grid on;
            xlabel('b_1_-', 'fontweight', 'bold');
            ylabel('b_2_-', 'fontweight', 'bold');
            zlabel('b_3_-', 'fontweight', 'bold');
            if (numOfRandomVar == 1)
                title('x = b_1*t_1 + b_2*{t_1}^2');
            else
                title('x = b_1*t_1 + b_2*t_2');
            end
        end
        
        if(strcmpi(varargin{1}, 'pca') == 1)
            
            % Principle Component Analysis (PCA)
            % representation of input data in the principal component 
            % space, eigenvalues(principal component variances)
            [~, newCoords, eigenvalues] = pca(dataset');
           
            varargout{:} = eigenvalues;
            newCoords = newCoords';
            
            % plot of the generated dataset in 3-D and in the principal
            % component space
            subplot(1, 2, 1);
            plot3(dataset(1, :), dataset(2, :), dataset(3, :), '*', 'color', 'm', 'MarkerSize', 10);
            grid on;
            xlabel('b_1_-', 'fontweight', 'bold');
            ylabel('b_2_-', 'fontweight', 'bold');
            zlabel('b_3_-', 'fontweight', 'bold');
            if (numOfRandomVar == 1)
                title('x = b_1*t_1 + b_2*{t_1}^2');
            else
                title('x = b_1*t_1 + b_2*t_2');
            end
            
            subplot(1, 2, 2);
            plot(newCoords(1, :), newCoords(2, :), 'go', 'MarkerSize', 10);
            grid on;
            xlabel('PC_1', 'fontweight', 'bold');
            ylabel('PC_2', 'fontweight', 'bold');
            if (numOfRandomVar == 1)
                title('after PCA');
            else
                title('after PCA');
            end
            
        end
        
    elseif(length(varargin) > 1) % only two options: either 'plotDataset' or 'pca'
        
        error('Invalid number of optional input arguments');
        
    end
    

end


end