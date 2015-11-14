        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise 2  %%
        %%                    Tasks A, B                  %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function tests the trained pattern recognition network returned  %
% by the function "trainPatternnet" on an unseen dataset, for example,  %
% digittest_dataset. It prints false alarm and detection probabilities. %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% trainedNet: previously trained pattern recognition network            %
% testInputs: test inputs/data                                          % 
% testLabels: test labels for testInputs                                %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% probabilities:  false alarm and  detection probabilities  (%)         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function  probabilities = testPatternnet(trainedNet, testInputs, testLabels)

% test network on test dataset (unseen data)
outputs = trainedNet(testInputs);

[~, ~, ~, probs] = confusion(testLabels, outputs);

fprintf('----------> Test Dataset <----------\n');
fprintf('False alarm probability: %.2f%%', probs(1, 2)*100);
fprintf('\n');
fprintf('Detection probability: %.2f%%', probs(2, 3)*100);
fprintf('\n');

probabilities = [probs(1, 2)*100  probs(2, 3)*100];

end