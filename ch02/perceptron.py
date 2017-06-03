import numpy
class Perceptron(object):
    """Perceptron classifier.

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

    def __init__(self, learn_rate=0.01, epochs=10):
        self.learn_rate = learn_rate
        self.epochs = epochs

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [sample_count, feature_count]
            Training vectors, defining the number of samples and features.
        y : array-like, shape = [sample_count]
            Target values.

        Returns
        -------
        self : object
        """
        self.fitted_weights_ = numpy.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learn_rate * (target - self.predict(xi))
                self.fitted_weights_[1:] += update * xi
                self.fitted_weights_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate new input"""
        return numpy.dot(X, self.fitted_weights_[1:]) + self.fitted_weights_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return numpy.where(self.net_input(X) >= 0.0, 1, -1)
