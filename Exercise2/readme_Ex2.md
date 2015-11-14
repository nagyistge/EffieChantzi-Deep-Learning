## Exercise 2

It contains all the .m files needed for tasks A, B and C. <br />
When the issue with task C(ii) will be solved, it will be updated with .m files for tasks D, E.

### `Tasks A, B`

> Evaluate Performance Classification Once

* 5 neurons and 'crossentropy'
```probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, [1 5000], 5);```
* 5 neurons and 'mse'
``` probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, [1 5000], 5, 'mse');```
* 10 neurons and 'crossentropy'
```probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, [1 5000], 10);```
* 10 neurons and 'mse'
``` probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, [1 5000], 10, 'mse');```
* 25 neurons and 'crossentropy'
```probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, [1 5000], 25);```
* 25 neurons and 'mse'
``` probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, [1 5000], 25, 'mse');```
* [25 10] (2-hidden layer) and 'crossentropy'
``` probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, [1 5000], [25 10]);```
* [25 10] (2-hidden layer) and 'mse'
``` probabilities = testClassifierOnce(XTrain, LTrain, XTest, LTest, [1 5000], [25 10], 'mse');```


> Evaluate Performance Classification Multiple Times 
Run the following in the command prompt and a menu will guide you through
``` evaluateClassifier ```
