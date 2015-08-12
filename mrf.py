#! /usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import unicode_literals
from collections import defaultdict

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
np.random.seed(1)


class MRF(object):
    def __init__(self, input, theta=0.3, threshold=0.1):
        self.input = input
        self.shape = self.input.shape
        self.theta = theta
        self.threshold = threshold

        self.visible = nx.grid_2d_graph(self.shape[0], self.shape[1])
        self.hidden = nx.grid_2d_graph(self.shape[0], self.shape[1])

        for n in self.nodes():
            self.visible[n]['value'] = self.input[n[0], n[1]]
            f = lambda: np.array([1.0, 1.0])
            self.hidden[n]['messages'] = defaultdict(f)

    def nodes(self):
        for row in range(self.shape[0]):
            for column in range(self.shape[1]):
                yield (row, column)

    def prob(self, value):
        base = np.array([1 + self.theta if value == 0 else 1 - self.theta,
                         1 + self.theta if value == 1 else 1 - self.theta])
        return base

    def marginal(self, source, target):
        m = np.array([0.0, 0.0])
        for i in xrange(2):
            prob = self.prob(i)
            neighbors = self.hidden[source]['messages'].keys()
            for n in [n for n in neighbors if n != target]:
                prob *= self.hidden[source]['messages'][n]
            m[i] = np.sum(prob)
        return m

    def send_message(self, source):
        # sourceの周りのノードを取得
        targets = [n for n in self.hidden[source] if isinstance(n, tuple)]

        diff = 0
        for target in targets:
            message = self.marginal(source, target)
            message /= np.sum(message)
            messages = self.hidden[target]['messages']
            diff += np.sum(np.abs(messages[source] - message))
            messages[source] = message

        return diff

    def belief_propagation(self, loop=20):
        edges = [edge for edge in self.hidden.edges()]
        edges = [edge for edge in edges
                 if isinstance(edge[0], tuple) and isinstance(edge[1], tuple)]
        threshold = self.threshold * len(edges)

        # 観測値からの周辺分布をノードに送る
        for n in self.nodes():
            message = self.prob(self.visible[n]['value'])
            message /= np.sum(message)
            self.hidden[n]['messages'][n] = message
        yield

        for i in xrange(loop):
            diff = 0
            for n in self.nodes():
                diff += self.send_message(n)
                yield

            if diff < threshold:
                break

    @property
    def denoised(self):
        for p in self.belief_propagation():
            pass

        denoised = np.copy(self.input)
        for row, column in self.nodes():
            prob = np.array([1.0, 1.0])
            messages = self.hidden[(row, column)]['messages']
            for value in messages.values():
                prob *= value
            denoised[row, column] = 0 if prob[0] > prob[1] else 1
        return denoised


def get_corrupted_input(img, corruption_level):
    corrupted = np.copy(img)
    inv = np.random.binomial(n=1, p=corruption_level, size=img.shape)
    for row in xrange(img.shape[0]):
        for column in xrange(img.shape[1]):
            if inv[row, column]:
                corrupted[row, column] = ~(corrupted[row, column].astype(bool))
    return corrupted


def main():
    print '... get mnist data'
    mnist = fetch_mldata('MNIST original', data_home='.')

    fig, axes = plt.subplots(5, 3, figsize=(6, 8))

    data = mnist.data[[0, 7000, 14000, 21000, 28000]]

    print '... start training'
    for i, (axrow, img) in enumerate(zip(axes, data)):
        img = img.reshape(28, 28)
        img = (img >= 128).astype(int)

        corrupted = get_corrupted_input(img, 0.05)
        mrf = MRF(corrupted)

        if i == 0:
            axes[i][0].set_title('元画像')
            axes[i][1].set_title('ノイズあり')
            axes[i][2].set_title('ノイズ除去')
        axes[i][0].imshow(img, cmap=cm.Greys_r)
        axes[i][1].imshow(corrupted, cmap=cm.Greys_r)
        axes[i][2].imshow(mrf.denoised, cmap=cm.Greys_r)
        for ax in axrow:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    plt.show()




if __name__ == '__main__':
    main()

# End of Line.
