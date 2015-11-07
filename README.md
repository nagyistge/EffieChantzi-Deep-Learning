# Deep-Learning

Repository for the project "Deep Learning" in "Complex Data: Analysis & Visualization" course.

## Exercise 1

It contains all the .m files needed for tasks A, B, C and D. 
Below are the required commands in the command promt for all tasks.

### Task A

$ help nndatasets

--function fitting (regression)

$ help simplefit_dataset
$ [inputs, targets] = simplefit_dataset;
$ ClassificationRegressionNN(inputs, targets, 'R');
or 
$ nftool

--pattern recognition (classification)
$ help cancer_dataset
$ [inputs, targets] = cancer_dataset;
$ ClassificationRegressionNN(inputs, targets, 'C');
or
$ options = {'neurons', 20};
$ ClassificationRegressionNN(inputs, targets, 'C', options);
or 
$ nprtool

