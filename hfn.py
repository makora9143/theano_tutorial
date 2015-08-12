#! /usr/bin/env python
# -*- coding: utf-8 -*-

# divisionを呼ぶことで割算で一方を小数にしなくても計算結果が小数になる
from __future__ import unicode_literals, division

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import fetch_mldata
import numpy as np
np.random.seed(1)


class HopFieldNetwork(object):
    def __init__(self):
        self.threshold = 0

    def fit(self, train_list):
        self.dim = len(train_list[0])

        n = len(train_list)

        self.W = np.zeros((self.dim, self.dim))

        # 全てのデータの平均
        rho = np.sum([np.sum(t) for t in train_list]) / (n * self.dim)

        for m in xrange(n):
            t = train_list[m] - rho
            self.W += np.outer(t, t)
        for i in xrange(self.dim):
            self.W[i, i] = 0
        self.W /= n

    def predict(self, data, threshold=0, loop=10):
        self.threshold = threshold
        return [self._predict(d, loop=loop) for d in data]

    def _predict(self, xr, loop=10):
        e = self.energy(xr)
        for i in xrange(loop):
            xr = np.sign(np.dot(self.W, xr) - self.threshold)
            e_new = self.energy(xr)
            if e_new == e:
                return xr
            e = e_new
        return xr

    def energy(self, xr):
        return -np.dot(np.dot(xr, self.W), xr) + np.sum(xr * self.threshold)

    def plot_data(self, ax, data, with_energy=False):
        dim = int(np.sqrt(len(data)))
        assert dim * dim == len(data)

        img = (data.reshape(dim, dim) + 1) / 2
        ax.imshow(img, cmap=cm.Greys_r, interpolation='nearest')
        if with_energy:
            e = np.round(self.energy(data), 1)
            ax.text(0.95, 0.05, e,
                    color='r', ha='right', transform=ax.transAxes)
        return ax

    def plot_weight(self):
        fig, ax = plt.subplots(figsize=(5, 3))
        heatmap = ax.pcolor(self.W, cmap=cm.coolwarm)
        cbar = plt.colorbar(heatmap)

        ax.set_xlim(0, self.dim)
        ax.set_ylim(0, self.dim)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        return fig, ax



def get_corrupted_input(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted


def plot(hn, data, test, predicted, figsize=(5, 7)):
    fig, axes = plt.subplots(len(data), 3, figsize=figsize)
    for i, axrow in enumerate(axes):
        if i == 0:
            axrow[0].set_title('train data')
            axrow[1].set_title('input data')
            axrow[2].set_title('output data')
        hn.plot_data(axrow[0], data[i])
        hn.plot_data(axrow[1], test[i], with_energy=True)
        hn.plot_data(axrow[2], predicted[i], with_energy=True)

        for ax in axrow:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    return fig, axes


def preprocessing(input):
    input = (input >= np.max(input) / 2).astype(int)
    return input * 2 - 1


def main():
    data = [
        np.array([0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
                  0, 1, 1, 1, 0, 0, 1, 0, 1, 0]),
        np.array([1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0,
                  1, 0, 0, 1, 0, 1, 1, 1, 0, 0]),
        np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                  1, 0, 0, 0, 0, 0, 1, 1, 1, 0])
    ]

    mnist = fetch_mldata('MNIST original', data_home='.')
    data = mnist.data[[0, 7000, 14000, 21000, 28000, 35000]]

    # data = [d * 2 - 1 for d in data]
    data = [preprocessing(d) for d in data]

    hn = HopFieldNetwork()
    hn.fit(data)

    test = [get_corrupted_input(d, corruption_level=0.1) for d in data]
    predicted = hn.predict(test, threshold=48)

    plot(hn, data, test, predicted, figsize=(5, 5))
    plt.show()



if __name__ == '__main__':
    main()


# End of Line.
