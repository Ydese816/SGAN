'''
This script aims to get the top k shapelets
of targeted dataset. The interval of shapellets
will be stored in shapelet_pos dir.
'''

import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
from pyts.transformation import ShapeletTransform
from query_probability import load_ucr


def shapelet_transform(run_tag):
    # Shapelet transformation
    st = ShapeletTransform(n_shapelets=3, window_sizes=[0.1, 0.2, 0.3, 0.4
                                                        ], sort=True, verbose=1, n_jobs=14)
    path = 'data/' + run_tag + '/' + run_tag + '_eval.tsv'
    data = load_ucr(path)
    X = data[:100, 1:]
    print(X.shape)
    y = data[:100, 0]
    X_new = st.fit_transform(X, y)

    file = open('shapelet_pos/' + run_tag + '_shapelet_pos.txt', 'w+')
    for i, index in enumerate(st.indices_):
        idx, start, end = index
        file.write(run_tag + ' ')
        file.write(str(idx) + ' ')
        file.write(str(start) + ' ')
        file.write(str(end) + '\n')
    file.close()
    # Visualize the most discriminative shapelets
    # plt.figure(figsize=(6, 4))
    # for i, index in enumerate(st.indices_):
    #     idx, start, end = index
    #     plt.plot(X[idx], color='C{}'.format(i),
    #              label='Sample {}'.format(idx))
    #     plt.plot(np.arange(start, end), X[idx, start:end],
    #              lw=5, color='C{}'.format(i))
    #
    # plt.xlabel('Time', fontsize=12)
    # plt.title('The five more discriminative shapelets', fontsize=14)
    # plt.legend(loc='best', fontsize=8)
    # plt.savefig('shapelet_fig/' + run_tag + '_shapelet.pdf')
    # plt.show()


def plot_pdf(run_tag):
    path = 'data/' + run_tag + '/' + run_tag + '_eval.tsv'
    data = load_ucr(path)
    X = data[:, 1:]
    y = data[:, 0]
    pos_path = 'shapelet_pos/' + run_tag + '_shapelet_pos.txt'
    shapelet_pos = np.loadtxt(pos_path, usecols=(1, 2, 3))
    plt.figure(figsize=(6, 4))
    for i in range(3):
        idx, start, end = int(shapelet_pos[i, 0]), int(shapelet_pos[i, 1]), int(shapelet_pos[i, 2])
        plt.plot(X[idx], color='C{}'.format(i),
                 label='Sample {}'.format(idx))
        plt.plot(np.arange(start, end), X[idx, start:end],
                 lw=5, color='C{}'.format(i))

    plt.xlabel('%s'%run_tag, fontsize=12)
    plt.title('The top 3 shapelets', fontsize=14)
    plt.legend(loc=1, fontsize=8)
    plt.savefig('shapelet_fig/' + run_tag + '.png',pad_inches=0.1, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    name = ['ECG200']
    for n in name:
        shapelet_transform(n)
        plot_pdf(n)
