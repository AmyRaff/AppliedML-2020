
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, ward
import matplotlib.pyplot as plt
import os
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from helpers import iaml01cw2_helpers as helper

#<---- s1817812


# Q3.1
def iaml01cw2_q3_1():
    path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\data"
    Xtrn, Ytrn, Xtst, Ytst = helper.load_CoVoST2(path)
    km = KMeans(n_clusters=22, random_state=1)
    km.fit(Xtrn, Ytrn)
    inertia = km.inertia_
    print("Sum Squared Distance: ", inertia)
    labels = km.labels_
    label_freq = (Counter(labels))
    for key, value in label_freq.items():
        print("Language: ", key, ", Quantity: ", value)
# iaml01cw2_q3_1()


# Q3.2
def iaml01cw2_q3_2():
    path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\data"
    Xtrn, Ytrn, Xtst, Ytst = helper.load_CoVoST2(path)
    km = KMeans(n_clusters=22, random_state=1)
    l0, l1, l2, l3, l4, l5, l6, l7, l8, l9 = [], [], [], [], [], [], [], [], [], []
    l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21 = [], [], [], [], [], [], [], [], [], [], [], []
    languages = [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21]
    for row in range(Xtrn.shape[0]):
        for lang in range(0, len(languages)):
            if Ytrn[row] == lang:
                languages[lang].append(Xtrn[row])
    mean0, mean1, mean2, mean3 = np.mean(l0, axis=0), np.mean(l1, axis=0), np.mean(l2, axis=0), np.mean(l3, axis=0)
    mean4, mean5, mean6, mean7 = np.mean(l4, axis=0), np.mean(l5, axis=0), np.mean(l6, axis=0), np.mean(l7, axis=0)
    mean8, mean9, mean10 = np.mean(l8, axis=0), np.mean(l9, axis=0), np.mean(l10, axis=0)
    mean11, mean12, mean13 = np.mean(l11, axis=0), np.mean(l12, axis=0), np.mean(l13, axis=0)
    mean14, mean15, mean16 = np.mean(l14, axis=0), np.mean(l15, axis=0), np.mean(l16, axis=0)
    mean17, mean18, mean19 = np.mean(l17, axis=0), np.mean(l18, axis=0), np.mean(l19, axis=0)
    mean20, mean21 = np.mean(l20, axis=0), np.mean(l21, axis=0)
    means = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9, mean10, mean11, mean12, mean13,
             mean14, mean15, mean16, mean17, mean18, mean19, mean20, mean21]
    km = KMeans(n_clusters=22, random_state=1)
    km.fit(Xtrn, Ytrn)
    centers = km.cluster_centers_
    pca = PCA(2)
    pca.fit(means)
    new = pca.transform(means)
    pca1 = []
    for i in range(new.shape[0]):
        pca1.append(new[i, 0])
    pca2 = []
    for j in range(new.shape[0]):
        pca2.append(new[j, 1])
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    for x_, y_, label in zip(pca1, pca2, labels):
        ax.scatter([x_], [y_], label=label)
    plt.scatter(centers[:, 0], centers[:, 1], label="K Means Clusters", color="black")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Mean Vectors for Languages 0-21 in 2D-PCA plane, and KMeans Clusters for K=22")
    plt.legend(ncol=3)
    plt.savefig("3_2.png")
    plt.show()
# iaml01cw2_q3_2()


# Q3.3
def iaml01cw2_q3_3():
    path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\data"
    Xtrn, Ytrn, Xtst, Ytst = helper.load_CoVoST2(path)
    l0, l1, l2, l3, l4, l5, l6, l7, l8, l9 = [], [], [], [], [], [], [], [], [], []
    l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21 = [], [], [], [], [], [], [], [], [], [], [], []
    languages = [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21]
    for row in range(Xtrn.shape[0]):
        for lang in range(0, len(languages)):
            if Ytrn[row] == lang:
                languages[lang].append(Xtrn[row])
    mean0, mean1, mean2, mean3 = np.mean(l0, axis=0), np.mean(l1, axis=0), np.mean(l2, axis=0), np.mean(l3, axis=0)
    mean4, mean5, mean6, mean7 = np.mean(l4, axis=0), np.mean(l5, axis=0), np.mean(l6, axis=0), np.mean(l7, axis=0)
    mean8, mean9, mean10 = np.mean(l8, axis=0), np.mean(l9, axis=0), np.mean(l10, axis=0)
    mean11, mean12, mean13 = np.mean(l11, axis=0), np.mean(l12, axis=0), np.mean(l13, axis=0)
    mean14, mean15, mean16 = np.mean(l14, axis=0), np.mean(l15, axis=0), np.mean(l16, axis=0)
    mean17, mean18, mean19 = np.mean(l17, axis=0), np.mean(l18, axis=0), np.mean(l19, axis=0)
    mean20, mean21 = np.mean(l20, axis=0), np.mean(l21, axis=0)
    means = np.array([mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9, mean10, mean11, mean12, mean13,
             mean14, mean15, mean16, mean17, mean18, mean19, mean20, mean21])
    Z = linkage(means, 'ward')
    plt.figure(figsize=(10, 6))
    plt.xlabel("Distance")
    plt.ylabel("Language Number")
    plt.title("Heirarchical Clustering Dendrogram")
    dendrogram(Z, orientation="right")
    plt.savefig("3_3.png")
    plt.show()
# iaml01cw2_q3_3()


# Q3.4
def iaml01cw2_q3_4():
    path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\data"
    Xtrn, Ytrn, Xtst, Ytst = helper.load_CoVoST2(path)
    l0, l1, l2, l3, l4, l5, l6, l7, l8, l9 = [], [], [], [], [], [], [], [], [], []
    l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21 = [], [], [], [], [], [], [], [], [], [], [], []
    languages = [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21]
    for row in range(Xtrn.shape[0]):
        for lang in range(0, len(languages)):
            if Ytrn[row] == lang:
                languages[lang].append(Xtrn[row])
    output = []
    for language in languages:
        km = KMeans(n_clusters=3, random_state=1)
        km.fit(language)
        for k in range(3):
            output.append(km.cluster_centers_[k])
    Z1 = linkage(np.array(output), "ward")
    Z2 = linkage(np.array(output), "single")
    Z3 = linkage(np.array(output), "complete")
    plt.figure(figsize=(8, 7))
    plt.xlabel("Distance")
    plt.ylabel("Language Number")
    plt.title("Heirarchical Clustering Dendrogram for Ward Metric")
    dendrogram(Z1, orientation="right")
    plt.savefig("3_4_ward.png")
    plt.show()
    plt.figure(figsize=(8, 7))
    plt.xlabel("Distance")
    plt.ylabel("Language Number")
    plt.title("Heirarchical Clustering Dendrogram for Single Metric")
    dendrogram(Z2, orientation="right")
    plt.savefig("3_4_single.png")
    plt.show()
    plt.figure(figsize=(8, 7))
    plt.xlabel("Distance")
    plt.ylabel("Language Number")
    plt.title("Heirarchical Clustering Dendrogram for Complete Metric")
    dendrogram(Z3, orientation="right")
    plt.savefig("3_4_complete.png")
    plt.show()
# iaml01cw2_q3_4()


# Q3.5
def iaml01cw2_q3_5():
    path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\data"
    Xtrn, Ytrn, Xtst, Ytst = helper.load_CoVoST2(path)
    l0 = []
    for row in range(Xtrn.shape[0]):
        if Ytrn[row] == 0:
            l0.append(Xtrn[row])
    k_vals = [1, 3, 5, 10, 15]
    diag_scores = []
    for k in k_vals:
        gm = GaussianMixture(n_components=k, covariance_type="diag")
        gm.fit(l0)
        diag_scores.append(gm.score(l0))
    print("Diagonal: ", diag_scores)
    full_scores = []
    for k in k_vals:
        gm2 = GaussianMixture(n_components=k, covariance_type="full")
        gm2.fit(l0)
        full_scores.append(gm2.score(l0))
    print("Full: ", full_scores)
    plt.figure(figsize=(5, 5))
    plt.plot(k_vals, diag_scores, label="Diagonal-Covariance Matrix")
    plt.plot(k_vals, full_scores, label="Full-Covariance Matrix")
    plt.legend()
    plt.xlabel("Number of Mixture Components K")
    plt.ylabel("Per-Sample Average Log Likelihood")
    plt.title("Per-Sample Avg Log Likelihood vs Number of Components")
    plt.savefig("3_5.png")
    plt.show()
# iaml01cw2_q3_5()

