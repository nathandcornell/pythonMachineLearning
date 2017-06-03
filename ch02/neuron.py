import abc
import numpy

class Neuron(object):
    """Neuron base class.

    Parameters
    ----------
    learn_rate : float
        Learning rate (between 0.0 and 1.0)
    epochs : int
        Number of passes over the training dataset

    Attributes
    ----------
    fitted_weights_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in each epoch (pass)
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, learn_rate=0.01, epochs=50):
        self.learn_rate = learn_rate
        self.epochs = epochs

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit training data"""

    def net_input(self, X):
        """Calculate new input"""
        return numpy.dot(X, self.fitted_weights_[1:]) + self.fitted_weights_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return numpy.where(self.activation(X) >= 0.0, 1, -1)
