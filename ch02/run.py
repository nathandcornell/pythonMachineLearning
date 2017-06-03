#! python3

import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap
import numpy
import pandas
import perceptron as pptn

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(numpy.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = numpy.meshgrid(numpy.arange(x1_min, x1_max, resolution),
                              numpy.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(numpy.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plot.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plot.xlim(xx1.min(), xx1.max())
    plot.ylim(xx2.min(), xx2.max())

    # plot class samples
    for index, clss in enumerate(numpy.unique(y)):
        plot.scatter(x=X[y == clss, 0], y=X[y == clss, 1], alpha=0.8,
                     c=cmap(index), marker=markers[index], label=clss)

def main():
    data_file = pandas.read_csv('iris.data', header=None)

    # Get the first 100 rows (assuming the last 50 reserved for cross 
    # validation):
    y = data_file.iloc[0:100, 4].values
    # Convert the labels to integers:
    y = numpy.where(y == 'Iris-setosa', -1, 1)

    X = data_file.iloc[0:100, [0, 2]].values
    # # Create a scatter plot:
    # plot.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    # plot.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', 
    #              label='versicolor')
    # plot.xlabel('sepal length')
    # plot.ylabel('petal length')
    # plot.legend(loc="upper left")
    # plot.show()

    # Train the perceptron on the Iris data:
    perceptron = pptn.Perceptron(learn_rate=0.1, epochs=10)
    perceptron.fit(X, y)

    # # Plot the misclassification error on each epoch:
    # plot.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, 
    #           marker='o')
    # plot.xlabel('Epochs')
    # plot.ylabel('Number of Misclassifications')
    # plot.show()

    plot_decision_regions(X, y, classifier=perceptron)
    plot.xlabel('sepal length [cm]')
    plot.ylabel('petal length [cm]')
    plot.legend(loc='upper left')
    plot.show()

if __name__ == '__main__':
    main()
