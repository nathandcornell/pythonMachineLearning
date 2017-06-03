#! python3

import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap
import numpy
import pandas
import adalinegd

def main():
    data_file = pandas.read_csv('iris.data', header=None)

    y = data_file.iloc[0:100, 4].values
    y = numpy.where(y == 'Iris-setosa', -1, 1)
    X = data_file.iloc[0:100, [0, 2]].values

    figure, axes = plot.subplots(nrows=1, ncols=2, figsize=(8, 4))

    adaline = adalinegd.AdalineGD(epochs=50, learn_rate=0.01).fit(X, y)
    axes[0].plot(range(1, len(adaline.cost_) + 1), numpy.log10(adaline.cost_),
                 marker='o')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('log(Sum-squared-error)')
    axes[0].set_title('Adaline - Learn rate 0.01')

    adaline2 = adalinegd.AdalineGD(epochs=50, learn_rate=0.0001).fit(X, y)
    axes[1].plot(range(1, len(adaline2.cost_) + 1), adaline2.cost_, marker='o')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Sum-squared-error')
    axes[1].set_title('Adaline - Learning rate 0.0001')

    plot.tight_layout()
    plot.show()


if __name__ == '__main__':
    main()
