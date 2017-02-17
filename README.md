# Gnarl - Generally nice and reasonable learning
Welcome to Gnarl, an easy-to-use [deep learning](https://en.wikipedia.org/wiki/Deep_learning) framework for Python. Gnarl was created with the aim to help newcomers  build an understanding of how shallow and deep neural networks these machine learning methods work, what they do under the hood, and what they can be used for.

## Usage
Gnarl is designed to be as easy as possible to get started with. The following is a brief tutorial that showcases how to create deep neural networks using Gnarl.

### Import Gnarl


```python
# Import Gnarl model
from gnarl import Gnarl
```

### Instantiate a Gnarl model
When instantiating Gnarl, input and output data aren't strictly required, but dummy arrays that provide the correct number of input features and output is necessary in order to build create the hidden layers behind the scenes.

Let's first import some utilities and example data, and split the data set into training, validation and test sets. We are also going to normalize the input feature data, in order to avoid potential numerical instabilities and computational overflows that could otherwise occur.


```python
# Import some utilities
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load the Boston housing prices from scikit-learn
data = load_boston()

# Set data input and output
X = data['data']
y = data['target']

# Normalize all input data to have mean 0 and standard deviation 1
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Split data into training, validation and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25)
```

Now that we have some regression data to play around with, instantiating the model is easy. You can either specify individual hyperparameters of a Gnarl model, or specify them as a dictionary object and pass this to the constructor.


```python
# Specify hyperparameters for the model
nn_options = {
    'activation': 'leaky_relu',
    'learning_rate': 1e-4,
    'regularization': 0.,
    'verbose': True,
    'loss': 'mse',
    'batch_size': 10
}

# Instantiate model
gnarl = Gnarl(X_train, y_train, **nn_options)
```

### Add hidden layers


```python
# Add first hidden layer
gnarl.add_layer(10, activation='leaky_relu')

# Add second hidden layer
gnarl.add_layer(5, activation='leaky_relu')

# Add output layer
gnarl.add_layer(1, activation='none')
```

### Connect the layers
Finally, before we can start training the model using our Boston housing price data, we need to connect and build the computational graph that does all of the heavy-lifting behind the scenes.


```python
# Connect the layers
gnarl.connect_layers()
```

### Resulting model object
Once the nodes in the graph have been connected, the gnarl model will have the following parameters and methods (clipped out defaults for brevity):

```
['X',
 '_biases',
 '_init',
 '_input_layer',
 '_log_probs',
 '_probs',
 '_reset_graph',
 '_update_input',
 '_update_output',
 '_weights',
 'activation',
 'batch_size',
 'connect_layers',
 'fit',
 'graph',
 'add_layer',
 'layers_list',
 'learning_rate',
 'loss',
 'nodes',
 'predict',
 'trainables',
 'verbose',
 'y']
```

> **Note:** properties prepended with an underscore are used internally, and shouldn't be accessed from outside the model.

### Train the model
Gnarl supports [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) and will automatically shuffle the data behind the scenes to make sure that the batch data is drawn from a random sample of the training data. The `epochs` argument determines how many times the training procedure will sample new batch data to train the network on. The `fit()` method also specifies an optional boolean argument `fit_more_data` that makes it easy to continue training an existing model.

> **Note:** If you've set the optional argument `verbose=True`, you'll be able to see the network being trained.


```python
# Train the model by sampling from all training data
gnarl.fit(X_train, y_train, solver='sgd', epochs=500)
```

    Training model...
    Solver: sgd
    Total number of samples: 303
    Steps per epoch: 30
    ================================================================================
    Epoch: 1, Loss: 336.223
    Epoch: 500, Loss: 6.7422
    ================================================================================
    Finished training model.
    Final loss: 9.460


### Predict outputs
Finally, once you've trained the model, and validated the hyperparameters using the validation data set `X_valid` and `y_valid` (not done here), we can predict Boston house prices and compute an $R^2$ score to see how well our model captures the variation in the data.


```python
# Predict outputs
y_pred = gnarl.predict(X_test)

# Compute R^2 score
r2 = r2_score(y_test, y_pred)

# Print the score
print('The R-squared score is: %.2f' % r2)
```

    The R-squared score is: 0.85


## Classification
Gnarl supports multinomial classification using one-hot encoded labels. Any target class data must be transformed from a one-dimensional vector with unique class or target labels into a one-hot encoded matrix with dimensions `M x K` where `M` is the number of rows (training examples) and `K` is the number of class or target labels.

Predictions during multi-class classification using Gnarl can return two types of results. First, predictions can be returned as a matrix of probabilities with the same dimensions as the test label data. Or, you can return the predictions as a one-dimensional vector of predicted labels (from `0` and counting upwards as integers). The latter is returned by default.


```python
# Import one-hot encoder from sklearn
from sklearn.preprocessing import OneHotEncoder

# Let's import the Iris data from sklearn
from sklearn.datasets import load_iris
data = load_iris()

# Original labels contained in a one-dimensional vector
X_iris = data.data
y_iris = data.target
print('Original label vector:\n', y_iris)
print('Dimension of label vector:\n', y_iris.shape)
```

    Original label vector:
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     2 2]
    Dimension of label vector:
     (150,)


Using the `OneHotEncoder`, we can transform the `y_labels` vector into the appropriate form:


```python
# Instantiate an encoder
enc = OneHotEncoder()

# Fit and transform using the raw label vector y_labels
y_ohe_iris = enc.fit_transform(y_iris.reshape(-1,1)).toarray().astype(int)

# Split data - skipping validation split
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_ohe_iris,
                                                                        test_size=0.2)

# Print to confirm. Only first five samples shown for brevity.
print('M by K one-hot encoded label matrix:\n', y_train_iris[:5])
print('Dimension of one-hot encoded label matrix:\n', y_train_iris.shape)
```

    M by K one-hot encoded label matrix:
     [[0 0 1]
     [0 1 0]
     [0 1 0]
     [0 0 1]
     [1 0 0]]
    Dimension of one-hot encoded label matrix:
     (120, 3)


By default, `OneHotEncoder` returns a sparse matrix, so we first transform the new matrix into a dense representation, and finally perform a type cast to `int` as a safety precaution. This last part is not strictly necessary, but it is more intuitive to think of the target labels being represented as integers.

### Predict class labels
During multi-class classification, Gnarl provides an optional boolean argument `truncate_labels` that finds which labels have the highest estimated probability per test input data, and returns these as a one-dimensional vector. By default, this parameter is `True`.


```python
# Create simple model - using all default options
gnarl_iris = Gnarl(X_train_iris, y_train_iris,
                   loss='cross_entropy',
                   batch_size=5,
                   verbose=True)

# Add a hidden layer
gnarl_iris.add_layer(8, activation='leaky_relu')

# Add output layer - 3 output classes
gnarl_iris.add_layer(3, activation='none')

# Connect layers
gnarl_iris.connect_layers()

# Train model
gnarl_iris.fit(X_train_iris, y_train_iris, solver='sgd', epochs=500)

# Predict
y_pred_iris = gnarl_iris.predict(X_test_iris, truncate_labels=True)
```

    Training model...
    Solver: sgd
    Total number of samples: 120
    Steps per epoch: 24
    ================================================================================
    Epoch: 1, Loss: 8.374
    Epoch: 500, Loss: 0.247
    ================================================================================
    Finished training model.
    Final loss: 0.386


The predicted class labels, when using the argument `truncate_labels=True` yields:


```python
print(y_pred_iris)
print(y_pred_iris.shape)
```

    [0 1 1 1 2 0 0 0 2 2 2 1 2 2 1 2 0 1 0 2 0 1 0 2 0 0 0 1 2 0]
    (30,)


While, if we instead don't truncate the predictions, we get:


```python
# predict again
y_pred_iris_probs = gnarl_iris.predict(X_test_iris, truncate_labels=False)

print(y_pred_iris_probs[:5])
print(y_pred_iris_probs.shape)
```

    [[  9.99500885e-01   4.17782976e-04   8.13322324e-05]
     [  2.41863204e-03   9.76668442e-01   2.09129263e-02]
     [  4.12413440e-02   9.41335729e-01   1.74229266e-02]
     [  6.15161123e-04   8.51838043e-01   1.47546796e-01]
     [  2.05776614e-04   2.52665915e-02   9.74527632e-01]]
    (30, 3)


We can also double-check that the probabilities sum to one, just to be safe:


```python
print(np.sum(y_pred_iris_probs, axis=1, keepdims=True)[:5])
```

    [[ 1.]
     [ 1.]
     [ 1.]
     [ 1.]
     [ 1.]]


And finally, we can also quickly compute the accuracy score for the classification network to see how well the network performs (without any validation or tuning of hyperparameters):


```python
def accuracy_score(y, y_pred):
    return np.mean(y == y_pred) * 100

# Turn test labels into original one-dim vector
y_test_iris_trunc = np.argmax(y_test_iris, axis=1)

print('Accuracy is: %.2f percent' % accuracy_score(y_test_iris_trunc, y_pred_iris))
```

    Accuracy is: 96.67 percent


## Options
Gnarl has the following options that can be used to set the hyperparameters of a neural network model:

  * `activation: string` - Default is `'leaky_relu'`

    * `'sigmoid'`

    * `'relu'`

    * `'leaky_relu'`

    * `'none'` - Used to define output layer

  * `learning_rate: float` - Default is `1e-4`
  * `regularization: float` - Default is `0.`
  * `verbose: boolean` - Default is `False`
  * `loss: string` - Default is `'mse'`
    * `'mse'` - Mean Squared Error
    * `'cross_entropy'` - Cross-entropy error (loss). Uses multinomial log loss.
  * `batch_size: int` - Default is `10`, used during Stochastic Gradient Descent

## Saving and loading models
Gnarl also allows you to save and load your models for future use. The optional argument `path` allows you to specify where on disk to save the model. The default location is the same as the current directory. Pickle is the only file format currently supported.


```python
# Import save method
from Gnarl import save_model

# Save model
save_model(gnarl, model_name='gnarls_barkley', path='./')
```

To load a model, we do the following:


```python
# Import load method
from Gnarl import load_model

# Load model
gnarls_barkley = load_model('./gnarls_barkley.pickle')
```

## Author
_**Author:** Hugo Nordell_

_**Twitter:** @hugonordell_

_**License:** MIT_


```python

```
