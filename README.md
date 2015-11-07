# Deep-Learning

Repository for the project "Deep Learning" in "Complex Data: Analysis & Visualization" course.

## Exercise 1

It contains all the .m files needed for tasks A, B, C and D. <br />
Below are the required commands in the command prompt for all tasks.

### Task A

$ help nndatasets

#### Function Fitting (Regression) <br />
$ help simplefit_dataset <br />
$ [inputs, targets] = simplefit_dataset; <br />
$ ClassificationRegressionNN(inputs, targets, 'R'); <br />
or <br />
$ nftool 

#### Pattern Recognition (Classification) <br />
$ help cancer_dataset <br />
$ [inputs, targets] = cancer_dataset; <br />
$ ClassificationRegressionNN(inputs, targets, 'C'); <br />
or <br />
$ options = {'neurons', 20}; <br />
or <br />
$ options = {'division', [0.8 0.1 0.1]}; <br />
or <br />
$ options = {'neurons', 20, 'division', [0.8 0.1 0.1]} <br />
or other such combinations <br />
$ ClassificationRegressionNN(inputs, targets, 'C', options); <br />
or 
$ nprtool


### Task B

#### x = b1t1 + b2_t1_^2 

##### Plot Dataset
$ dataset = generateDataset(500, 3, 2, 1, 'modelfunc', 'plotDataset');






