# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from collections import defaultdict

import numpy as np
np.random.seed(1)
import networkx as nx

class MRF(object):

    def __init__(self, input, theta=0.3, threshold=0.1):
        # 入力画像
        self.input = input
        # 入力画像のサイズ
        self.shape = self.input.shape

        # ノイズを含む観測値
        self.visible = nx.grid_2d_graph(self.shape[0], self.shape[1])

        # ノイズ除去した真の値の推測値 (潜在変数)
        self.hidden = nx.grid_2d_graph(self.shape[0], self.shape[1])

        for n in self.nodes():
            # 入力画像の値を 各観測値ノードの 'value' にセット
            self.visible[n]['value'] = self.input[n[0], n[1]]

            # 潜在変数が隣接ノードから受け取るメッセージは
            # {ノード座標 : 各値 {0, 1} をとる確率} の辞書として保存
            f = lambda: np.array([1.0, 1.0])
            self.hidden[n]['messages'] = defaultdict(f)

        self.theta = theta
        self.threshold = threshold

    def nodes(self):
        # ノードの座標を (行番号, 列番号) の tuple として返す generator
        for r in range(self.shape[0]):
            for c in range(self.shape[1]):
                yield (r, c)

    @property
    def denoised(self):
        """
        ノイズ除去した画像を返す
        """
        # 確率伝播法をループ実行 (Loopy Belief Propagation)
        for p in self.belief_propagation():
            pass

        # ノイズ除去後の画像
        denoised = np.copy(self.input)
        for r, c in self.nodes():
            prob = np.array([1.0, 1.0])
            messages = self.hidden[(r, c)]['messages']
            for value in messages.values():
                prob *= value
            # 周辺分布から 潜在変数の推定値を算出
            denoised[r, c] = 0 if prob[0] > prob[1] else 1
        return denoised

    def send_message(self, source):
        """
        sourceで指定されたノードからから各隣接ノードへ
        メッセージを送信する
        """
        targets = [n for n in self.hidden[source] if isinstance(n, tuple)]

        # 収束判定のため、前回ループ時のメッセージとの差分をとる
        diff = 0
        for target in targets:
            # source で指定されたノードの周辺分布を求める
            message = self.marginal(source, target)
            message /= np.sum(message)
            messages = self.hidden[target]['messages']
            # 前回ループ時のメッセージとの差分を加算
            diff += np.sum(np.abs(messages[source] - message))
            messages[source] = message
        # 差分の総和を返す
        return diff

    def marginal(self, source, target):
        """
        source で指定されたノードの周辺確率を求める
        """
        m = np.array([0.0, 0.0])
        for i in range(2):
            prob = self.prob(i)
            neighbors = self.hidden[source]['messages'].keys()
            # メッセージ送信先である target ノードは周辺分布の計算から除外する
            for n in [n for n in neighbors if n != target]:
                prob *= self.hidden[source]['messages'][n]
            m[i] = np.sum(prob)
        return m

    def belief_propagation(self, loop=20):
        # 収束判定条件
        # ここでは グラフのエッジ数 * threshold で指定された数値とした
        edges = [e for e in self.hidden.edges()]
        edges = [e for e in edges if isinstance(e[0], tuple) and isinstance(e[1], tuple)]
        threshold = self.threshold * len(edges)

        for n in self.nodes():
            message = self.prob(self.visible[n]['value'])
            message /= np.sum(message)
            self.hidden[n]['messages'][n] = message
        yield

        for i in range(loop):
            diff = 0
            for n in self.nodes():
                diff += self.send_message(n)
                yield

            # 収束判定
            if diff < threshold:
                break

    def prob(self, value):
        """
        周辺分布を求める
        """
        base = np.array([1 + self.theta if value == 0 else 1 - self.theta,
                         1 + self.theta if value == 1 else 1 - self.theta])
        return base


def get_corrupted_input(img, corruption_level):
    """
    ノイズを加えた画像を生成
    """
    corrupted = np.copy(img)
    inv = np.random.binomial(n=1, p=corruption_level,
                             size=img.shape)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if inv[r, c]:
                corrupted[r, c] = ~(corrupted[r, c].astype(bool))
    return corrupted

# MNIST データをダウンロード / ロード
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home=".")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
fig, axes = plt.subplots(5, 3, figsize=(6, 8))

# サンプルデータをスライス
data = mnist.data[[0, 7000, 14000, 21000, 28000]]

for i, (axrow, img) in enumerate(zip(axes, data)):
    img = img.reshape(28, 28)
    # 2値画像に変換
    img = (img >= 128).astype(int)

    # 5% のノイズを付与
    corrupted = get_corrupted_input(img, 0.05)
    # ノイズ付与画像からマルコフ確率場インスタンスを生成
    mrf = MRF(corrupted)

    if i == 0:
        axes[i][0].set_title('元画像')
        axes[i][1].set_title('ノイズ付与')
        axes[i][2].set_title('推測値')
    axes[i][0].imshow(img, cmap=cm.Greys_r)
    axes[i][1].imshow(mrf.input, cmap=cm.Greys_r)
    # MRF.denoised プロパティはノイズ除去した推測値を返す
    axes[i][2].imshow(mrf.denoised, cmap=cm.Greys_r)
    for ax in axrow:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
plt.show()
