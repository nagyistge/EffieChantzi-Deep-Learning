## Exercise 3

It contains all the .m files needed for tasks A, B, C, D, E, F, G. <br />

### `Task A`

> Evaluate Performance Classification Once

* 5 neurons and 'crossentropy'
```
probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, [1 5000], 5);
```
* 5 neurons and 'mse'
```
probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, [1 5000], 5, 'mse');
```
* 10 neurons and 'crossentropy'
```
probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, [1 5000], 10);
```
* 10 neurons and 'mse'
```
probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, [1 5000], 10, 'mse');
```
* 25 neurons and 'crossentropy'
```
probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, [1 5000], 25);
```
* 25 neurons and 'mse'
```
probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, [1 5000], 25, 'mse');
```
* {25-10} (2 hidden layers) and 'crossentropy'
```
probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, [1 5000], [25 10]);
```
* {25-10} (2 hidden layers) and 'mse'
``` 
probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, [1 5000], [25 10], 'mse');
```


> Evaluate Performance Classification Multiple Times 

Run the following in the command prompt and a menu will guide you through
``` 
evaluateClassifier 
```

### `Task C`
Run the following in the command prompt and a menu will guide you through
``` 
SAE
```


### `Tasks D, E`
expected soon..




