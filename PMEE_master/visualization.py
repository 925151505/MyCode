'''
Visualization of experimental performance
'''


import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pickle
#For T-SNE
with open(f"./feature//PMEE_mosi.pkl", "rb") as handle:
    data = pickle.load(handle)
    X = np.array(data['data'])
    y = np.array(data['label'])
    # print(len(X))
    # print(len(y))
    tsne = TSNE(n_components=2, perplexity=60).fit_transform(X)
    aa = tsne[:, 0]
    bb = tsne[:, 1]
    #color = ['limegreen', 'cornflowerblue', 'orange', 'red', 'yellow', 'blue']
    #Blue represents negativity, gray represents positivity
    color = ['blue', 'gray']
    for i in range(tsne.shape[0]):
        plt.scatter(aa[i], bb[i], facecolor=color[y[i]], alpha=0.7)
    #plt.savefig('image/MAG_xlnet_mosi_tsne.png', transparent=True, dpi=1800)
    plt.show()

#For heatmap
# trans_mat = np.array([[319, 38],
#                      [35, 221]], dtype=int)
# trans_mat = np.array([[1083, 250],
#                       [219, 2039]], dtype=int)
#
# trans_prob_mat = (trans_mat.T / np.sum(trans_mat, 1)).T
#
# if True:
#     label = ['positive', 'negative']
#     df = pd.DataFrame(trans_prob_mat, index=label, columns=label)
#
#     # Plot
#     plt.figure(figsize=(7.5, 6.3))
#     ax = sns.heatmap(df, xticklabels=df.corr().columns,
#                      yticklabels=df.corr().columns, cmap='magma',
#                      linewidths=6, annot=True)
#
#     # Decorations
#     plt.xticks(fontsize=16, family='Times New Roman')
#     plt.yticks(fontsize=16, family='Times New Roman')
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#     plt.tight_layout()
#     plt.savefig('image/mosei_heatmap.png', transparent=True, dpi=1800)
#     plt.show()