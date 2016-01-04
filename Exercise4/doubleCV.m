        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercise 4  %%
        %%                    Task B                      %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script pertains to the implementation of double cross-validation %
% on a subset of the initial digittrain_dataset, with limited risk of   %
% overfitting.                                                          %
% The size of the used subset of images is defined in line 41.          %    
% It is set to 1600 for 3 double cross-validations with 5 outer and 3   %
% inner folds. The choice of this number requires attention depending on%
% the current double cross-validation structure. This is why it is not  %
% requested as a user-defined input. For more info read the .m file     %
% 'createSubsetForCrossValidation.m'.                                   %
%                                                                       %
%                                                                       %
% Run this script and a menu will guide you through. More precisely,    % 
% the number of the double cross-validations for execution, the number  %
% of outer and inner folds, as well as the set of values for the second %
% hidden layer H used for the optimization as a row vector, are         %
% requested from the user.                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;
close all;

%% Load trainning dataset into memory

[XTrain, LTrain] = digittrain_dataset;

% transform the cell array of XTrain images into a matrix of vectors
images = createInputs(XTrain, [1 5000]);
dim = size(images, 1);
obs = size(images, 2);

% transform label data of train images into the proper format
labels = createNewLabels(LTrain, [1 5000]);

% independent test set
[XTest, LTest] = digittest_dataset;
imagesTest = createInputs(XTest, [1 5000]);

%% New Training dataset consisting only of 1600 samples: 493 "3" & (obs_subset-493) "other" examples
obs_subset = 1600;
images_subset = createSubsetForCrossValidation(images, labels, obs_subset);

%% Request user-defined parameters for the double CV(s) procedure(s)

times_doubleCV = 0;
while ((sum(mod(times_doubleCV, 1)) > 0) || (sum(times_doubleCV < 1) > 0) || (isempty(times_doubleCV)))

    times_doubleCV = input('Enter the number of times for the double CV procedure: \n');
    
end

% the less the number of the outer CV folds, the bigger the number of the
% subset images must be, so that the coefficient matrix of PCA is of full rank
K_outer = 0;
while ((sum(mod(K_outer, 1)) > 0) || (sum(K_outer < 1) > 0) || (isempty(K_outer)))

    K_outer = input('Enter the number of outer CV folds: \n');
    
end

K_inner = 0;
while ((sum(mod(K_inner, 1)) > 0) || (sum(K_inner < 1) > 0) || (isempty(K_inner)))

    K_inner = input('Enter the number of inner CV folds: \n');
    
end

H = 0;
while ((sum(mod(H, 1)) > 0) || (sum(H < 1) > 0) || (isempty(H)))

    H = input('Enter the bottleneck sizes as a row vector: \n');
    
end
len = length(H);

%% additional variables

% initialization of error estimates
E_inner_avg = zeros((times_doubleCV*K_outer), len); 
E_inner_avg_star = zeros(times_doubleCV, K_outer);
E_outer = zeros(times_doubleCV, K_outer);
E_outer_test = zeros(times_doubleCV, K_outer);


count1 = 0;
time_start = tic;

%% Double CV loop
for i_dcv = 1 : times_doubleCV
   
    
    % indices of outer CV partitions
    ind_outer = crossvalind('Kfold', obs_subset, K_outer);
    
    for i_outer = 1 : K_outer

        testing_outer = (ind_outer == i_outer);
        training_outer = ~testing_outer;

        train_images_outer = images_subset(:, training_outer);
        test_images_outer = images_subset(:, testing_outer);

        % indices of inner CV partitions
        ind = crossvalind('Kfold', size(train_images_outer, 2), K_inner);

        E_inner = zeros(K_inner, len);


        %% 3-fold CV inner loop
        for i_inner = 1 : K_inner

            testing_inner = (ind == i_inner);    % indices for the fold used for evaluation
            training_inner = ~testing_inner;     % indices for the (K - 1) folds used for training

            train_images_inner = train_images_outer(:, training_inner);
            test_images_inner = train_images_outer(:, testing_inner);    

            fprintf(' ------------------->  Double CV: %d of %d - Outer: Iter %d of %d - Inner: Iter %d of %d <------------------- \n', i_dcv, times_doubleCV, i_outer, K_outer, i_inner, K_inner);
           
            for h = 1 : length(H)

                fprintf('H = %d ...\n', H(h));
                fprintf('\n');

                [W1, W2, W3, W4, b1, b2, b3, b4] = PCAInitialization([90 H(h)], 1/40, train_images_inner, 'logsig');

                %% Weights and biases initialization 

                architectureVector = [dim 90 H(h) 90 dim];
                net = customizeNetwork(architectureVector, W1, W2, W3, W4, b1, b2, b3, b4, 'logsig', 'cv');

                %% Training

                net.trainParam.epochs = 700;
                net.trainParam.min_grad = 10^-4;
                net = train(net, train_images_inner, train_images_inner);

                %% Testing on the Kth inner fold

                outputs_K_inner = net(test_images_inner);
                E_inner(i_inner, h) = estimateTotalMSE(test_images_inner, outputs_K_inner);
                

            end
            

        end

        E_inner_avg((i_outer + count1), :) = mean(E_inner);
        [r, c] = find(E_inner_avg((i_outer + count1), :) == min(E_inner_avg((i_outer + count1), :)));
        E_inner_avg_star(i_dcv, i_outer) = min(E_inner_avg((i_outer + count1), :));

        [W1, W2, W3, W4, b1, b2, b3, b4] = PCAInitialization([90 H(c)], 1/40, train_images_outer, 'logsig');

        %% Weights and biases initialization 

        architectureVector = [dim 90 H(c) 90 dim];
        net = customizeNetwork(architectureVector, W1, W2, W3, W4, b1, b2, b3, b4, 'logsig', 'cv');

        %% Training

        net.trainParam.epochs = 700;
        net.trainParam.min_grad = 10^-4;
        net = train(net, train_images_outer, train_images_outer);

        %% Testing on Kth outer fold

        outputs_K_outer = net(test_images_outer);
        E_outer(i_dcv, i_outer) = estimateTotalMSE(test_images_outer, outputs_K_outer);

        %% Test Network on Test Dataset
        outputs_K_outer_test = net(imagesTest);
        E_outer_test(i_dcv, i_outer) = estimateTotalMSE(imagesTest, outputs_K_outer_test);
        plotComparison(imagesTest, outputs_K_outer_test, [1 5], architectureVector, E_outer_test(1, i_outer), 'cv');
        saveas(gcf, sprintf('TestDataset_K%d_dCV%d.png', i_outer, i_dcv));


    end

    count1 = count1 + K_outer;
    
end

time_end = toc(time_start);

% time required for the double CV(s)
t = datevec(time_end./(60*60*24));
if (times_doubleCV == 1)

    fprintf('\n Time elapsed for %d double CV with inner = %d and outer = %d folds: %.2f (hrs) %.2f (min) %.2f (sec)\n', times_doubleCV, K_inner, K_outer, t(4), t(5), t(6));
    
else
    
    fprintf('\n Time elapsed for %d double CVs with inner = %d and outer = %d folds: %.2f (hrs) %.2f (min) %.2f (sec)\n', times_doubleCV, K_inner, K_outer, t(4), t(5), t(6));
    
end

%% averaging error estimates from the different times that double CV is performed (times_doubleCV)
mean_E_inner_avg_star = zeros(1, times_doubleCV);
for i = 1 : times_doubleCV
    
    mean_E_inner_avg_star(1, i) = mean(E_inner_avg_star(i, :));
    
end

mean_E_outer = zeros(1, times_doubleCV);
for i = 1 : times_doubleCV
    
    mean_E_outer(1, i) = mean(E_outer(i, :));
    
end

mean_E_outer_test = zeros(1, times_doubleCV);
for i = 1 : times_doubleCV
    
    mean_E_outer_test(1, i) = mean(E_outer_test(i, :));
    
end


%% standard deviations of above error estimates, if one double CV is performed
if (times_doubleCV == 1)
    
    std_E_inner_avg_star = std(E_inner_avg_star);
    std_E_outer = std(E_outer);
    std_E_outer_test = std(E_outer_test);

%% final averages and standard deviations of above error estimates, if multiple double CVs are performed
else
   
    %E_inner_avg_star
    totalMean_E_inner_avg_star = mean(mean_E_inner_avg_star);
    std_E_inner_avg_star = std(mean_E_inner_avg_star);
    
    %E_outer
    totalMean_E_outer = mean(mean_E_outer);
    std_E_outer = std(mean_E_outer);
    
    %E_outer_test
    totalMean_E_outer_test = mean(mean_E_outer_test);
    std_E_outer_test = std(mean_E_outer_test);
    
end

