import numpy
import neuron

class AdalineGD(neuron.Neuron):
    """Adaline Gradient Descent classifier"""

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
        self.cost_ = []

        for i in range(self.epochs):
            net_input = self.net_input(X)
            output = self.activation(X)
            errors = (y - output)
            self.fitted_weights_[1:] += self.learn_rate * X.T.dot(errors)
            self.fitted_weights_[0] += self.learn_rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        return self
