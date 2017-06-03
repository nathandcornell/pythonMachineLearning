import numpy
import neuron

class Perceptron(neuron.Neuron):
    """Perceptron classifier"""

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
