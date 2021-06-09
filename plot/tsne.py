import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import numpy as np
plt.rc('font',family='Times New Roman')

color = ['b', 'r', 'g', 'b', 'k', 'm', 'c']
ave = lambda x: sum(x) / len(x)
embedding = np.load("deebert/sst2_layer_embedding.npy")
label = np.load("deebert/sst2_label_ids.npy")


def draw(E, ax, label, label_name, classes, ppl=50, layer_idx=1, task='sst2', n_iter=400):
    # pca=PCA(n_components=2)
    N = E.shape[0]
    E = TSNE(n_components=2, perplexity=ppl, n_iter=n_iter).fit_transform(E)
    # pca.fit(E)
    # E = pca.transform(E)
    x = [[] for i in range(classes)]
    y = [[] for i in range(classes)]
    for i in range(N):
        x[label[i]].append(float(E[i, 0]))
        y[label[i]].append(float(E[i, 1]))

    for i in range(classes):
        ax.scatter(x[i], y[i], color=color[i], s=1, alpha=0.6, label=label_name[i])  # 画出来散点位置
    # plts = []
    # for i in range(classes):
    #    plts.append(plt.scatter(ave(x[i]), ave(y[i]), color=color[i], s=5, alpha=1.0), label=label_name[i]) #画出来类别原型位置
    ax.legend(fontsize=12, markerscale=5., )
    ax.set_title(task)
    # ax.grid(alpha=0.3)
    # plt.savefig(path)
    # plt.cla()
    # plt.savefig("%s-%dL.pdf" %(task, layer_idx), dpi=400)
    # plt.show()

draw(complete_2l_sst2_embedding, label=complete_2l_sst2_label, label_name=['negative', 'positive'], classes=2,
    dirname='bert-complete-pdf', layer_idx=2 , task='sst2', ppl=30)