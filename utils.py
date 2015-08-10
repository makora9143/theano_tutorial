#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np


def scale_to_unit_interval(ndar, eps=1e-8):
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(
        X, img_shape, tile_shape,
        tile_spacing=(0, 0),
        scale_rows_to_unit_interval=True,
        output_pixel_vals=True):
    """
        :type X: numpy.array
        :param X: 入力行列．おそらく学習した重みを入力したりする

        :type img_shape: タプル （縦, 横）
        :param img_shape: 一つ一つの画像のサイズ．mnistとかだったら28×28

        :type tile_shape: タプル （縦, 横）
        :param tile_shape: img_shapeの画像をどう並べていくか．

        :type tile_spacing: タプル （縦, 横）
        :param tile_spacing: 並べていく時の画像間のスペース

    """

    # check element size
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # （画像サイズ ＋ 画像間）×並べる数ー画像間
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        if output_pixel_vals:
            out_array = np.zeros(
                (out_shape[0], out_shape[1], 4),
                dtype='uint8'
            )
            channel_defaults = [0, 0, 0, 255]
        else:
            out_array = np.zeros(
                (out_shape[0], out_shape[1], 4),
                dtype=X.dtype
            )
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]

            else:
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape,tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals
                )
        return out_array

    else:
        H, W = img_shape
        Hs, Ws = tile_spacing

        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                # 画像一つ一つを出力していく
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)

                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


# End of Line.
