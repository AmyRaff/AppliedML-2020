
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import pandas as pd
from templates import iaml01cw2_my_helpers as myhelper

# <---- s1817812


# Q1.1
def iaml01cw2_q1_1():
    Xtrn, Xtrn_nm, Ytrn, Xtst_nm, Ytst = myhelper.normalize_data()
    first = []
    last = []
    for i in range(4):
        first.append(Xtrn_nm[0,:][i])
        last.append(Xtrn_nm[-1,:][i])
    print("First 4 elements of first sample: ", first)
    print("First 4 elements of last sample: ", last)
# iaml01cw2_q1_1()


# Q1.2
def iaml01cw2_q1_2():
    Xtrn, Xtrn_nm, Ytrn, Xtst_nm, Ytst = myhelper.normalize_data()
    zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines = [], [], [], [], [], [], [], [], [], []
    classes = [zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines]
    for row in range(Xtrn.shape[0]):
        for cl in range(0, len(classes)):
            if Ytrn[row] == cl:
                classes[cl].append(Xtrn[row])
    images = []
    for x in [zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines]:
        images.append(myhelper.get_mean_img(x))
        images.append(myhelper.get_closest_image(x))
        images.append(myhelper.get_second_closest_image(x))
        images.append(myhelper.get_second_furthest_image(x))
        images.append(myhelper.get_furthest_image(x))
    ncol, nrow = 5, 10
    plt.figure(figsize=(1.2 * nrow, 4 * ncol))
    for i in range(nrow * ncol):
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(images[i], cmap="gray_r")
        plt.xticks(())
        plt.yticks(())
        plt.title("class: " + str(i // ncol))
    plt.savefig("1_2.png")
    plt.show()
# iaml01cw2_q1_2()


# Q1.3
def iaml01cw2_q1_3():
    Xtrn, Xtrn_nm, Ytrn, Xtst_nm, Ytst = myhelper.normalize_data()
    pca = PCA(5)
    pca.fit(Xtrn_nm)
    print("Variances: ", pca.explained_variance_)
# iaml01cw2_q1_3()


# Q1.4
def iaml01cw2_q1_4():
    Xtrn, Xtrn_nm, Ytrn, Xtst_nm, Ytst = myhelper.normalize_data()
    pca = PCA(784)
    pca.fit(Xtrn_nm)
    var = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 785), var)
    plt.xlabel("Number of Principal Components")
    plt.xticks(scipy.arange(0, 805, 50))
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Explained Variance Ratio for 1 - 784 Principal Components")
    plt.savefig("1_4.png")
    plt.show()
# iaml01cw2_q1_4()


# Q1.5
#def iaml01cw2_q1_5():
#    return 0
#iaml01cw2_q1_5()


# Q1.6
def iaml01cw2_q1_6():
    Xtrn, Xtrn_nm, Ytrn, Xtst_nm, Ytst = myhelper.normalize_data()
    zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines = [], [], [], [], [], [], [], [], [], []
    classes = [zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines]
    for row in range(Xtrn_nm.shape[0]):
        for cl in range(0, len(classes)):
            if Ytrn[row] == cl:
                classes[cl].append(Xtrn_nm[row])
    data = [zeros[0], ones[0], twos[0], threes[0], fours[0], fives[0], sixes[0], sevens[0], eights[0], nines[0]]
    mses = []
    for k in [5, 20, 50, 200]:
        pca = PCA(k)
        pca.fit(Xtrn_nm)
        pca_data = pca.transform(data)
        proj_data = pca.inverse_transform(pca_data)
        these_mses = []
        for i in range(proj_data.shape[0]):
            rmse = np.sqrt(mean_squared_error(data[i], proj_data[i]))
            these_mses.append(rmse)
        mses.append(these_mses)
    print("RMSEs for K = 5:\n", mses[0], "\nRMSEs for K = 20:\n", mses[1], "\nRMSEs for K = 50:\n", mses[2],
          "\nRMSEs for K = 200:\n", mses[3])
# iaml01cw2_q1_6()


# Q1.7
def iaml01cw2_q1_7():
    Xtrn, Xtrn_nm, Ytrn, Xtst_nm, Ytst = myhelper.normalize_data()
    Xmean = np.mean(Xtrn, axis=0)
    zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines = [], [], [], [], [], [], [], [], [], []
    classes = [zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines]
    for row in range(Xtrn_nm.shape[0]):
        for cl in range(0, len(classes)):
            if Ytrn[row] == cl:
                classes[cl].append(Xtrn_nm[row])
    data = [zeros[0], ones[0], twos[0], threes[0], fours[0], fives[0], sixes[0], sevens[0], eights[0], nines[0]]
    fives, twenties, fifties, twohundos = [], [], [], []
    for k in [5, 20, 50, 200]:
        pca = PCA(k)
        pca.fit(Xtrn_nm)
        pca_data = pca.transform(data)
        proj_data = pca.inverse_transform(pca_data)
        for i in range(len(proj_data)):
            if k == 5:
                fives.append(proj_data[i].reshape((28, 28)) + Xmean.reshape((28, 28)))
            if k == 20:
                twenties.append(proj_data[i].reshape((28, 28)) + Xmean.reshape((28, 28)))
            if k == 50:
                fifties.append(proj_data[i].reshape((28, 28)) + Xmean.reshape((28, 28)))
            if k == 200:
                twohundos.append(proj_data[i].reshape((28, 28)) + Xmean.reshape((28, 28)))
    images = []
    for i in range(len(fives)):
        images.append(fives[i])
        images.append(twenties[i])
        images.append(fifties[i])
        images.append(twohundos[i])
    ncol, nrow = 4, 10
    plt.figure(figsize=(1 * nrow, 4 * ncol))
    for i in range(nrow * ncol):
        plt.subplot(nrow, ncol, i + 1)
        plt.subplots_adjust(wspace=0, hspace=0.1)
        plt.imshow(images[i], cmap="gray_r")
        plt.xticks(())
        plt.yticks(())
    plt.savefig("1_7.png")
    plt.show()
# iaml01cw2_q1_7()


# Q1.8
def iaml01cw2_q1_8():
    Xtrn, Xtrn_nm, Ytrn, Xtst_nm, Ytst = myhelper.normalize_data()
    pca = PCA(2)
    pca.fit(Xtrn_nm)
    new = pca.transform(Xtrn_nm)
    pc1 = new[:, 0]
    pc2 = new[:, 1]
    df = pd.DataFrame()
    df['PC1'] = pc1
    df['PC2'] = pc2
    df['Class'] = Ytrn
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=df.PC1, y=df.PC2, hue=df.Class, cmap="coolwarm", legend="full")
    plt.title("Projection of Xtrn_nm onto 2D PCA Plane")
    plt.savefig("1_8.png")
    plt.show()
# iaml01cw2_q1_8()
