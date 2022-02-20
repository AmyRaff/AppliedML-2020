
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from templates import iaml01cw2_my_helpers as myhelper

# <---- s1817812


# Q2.1
def iaml01cw2_q2_1():
    Xtrn, Xtrn_nm, Ytrn, Xtst_nm, Ytst = myhelper.normalize_data()
    lm = LogisticRegression()
    lm.fit(Xtrn_nm, Ytrn)
    accuracy = lm.score(Xtst_nm, Ytst)
    print("Accuracy Score: ", accuracy)
    preds = lm.predict(Xtst_nm)
    cm = confusion_matrix(Ytst, preds)
    print("Confusion Matrix:\n", cm)
# iaml01cw2_q2_1()


# Q2.2
def iaml01cw2_q2_2():
    Xtrn, Xtrn_nm, Ytrn, Xtst_nm, Ytst = myhelper.normalize_data()
    cl = SVC()
    cl.fit(Xtrn_nm, Ytrn)
    accuracy = cl.score(Xtst_nm, Ytst)
    print("Accuracy Score: ", accuracy)
    preds = cl.predict(Xtst_nm)
    cm = confusion_matrix(Ytst, preds)
    print("Confusion Matrix:\n", cm)
# iaml01cw2_q2_2()


# Q2.3
def iaml01cw2_q2_3():
    Xtrn, Xtrn_nm, Ytrn, Xtst_nm, Ytst = myhelper.normalize_data()
    lr = LogisticRegression()
    pca = PCA(2)
    pca.fit(Xtrn_nm, Ytrn)
    X_train = pca.transform(Xtrn_nm)
    lr.fit(X_train, Ytrn)
    component1 = X_train[:, 0]
    component2 = X_train[:, 1]
    sigma1 = np.std(component1)
    sigma2 = np.std(component2)
    x1, x2 = np.meshgrid(np.arange(start=int((-5 * sigma1)), stop=int((5 * sigma2)), step=0.01),
                         np.arange(start=int((-5 * sigma1)), stop=int((5 * sigma2)), step=0.01))
    r1, r2 = x1.flatten(), x2.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1, r2))
    plt.contourf(x1, x2, lr.predict(grid).reshape(x1.shape), cmap="coolwarm")
    plt.xlim((-5 * sigma1), (5 * sigma2))
    plt.ylim((-5 * sigma1), (5 * sigma2))
    plt.colorbar(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Class")
    plt.title("Decision Regions spanned by 2D PCA Plane (Logistic Regression)")
    plt.xlabel("(-5 * std(PC1)) -> (5 * std(PC2))")
    plt.ylabel("(-5 * std(PC1)) -> (5 * std(PC2))")
    plt.savefig("2_3.png")
    plt.show()
# iaml01cw2_q2_3()


# Q2.4
def iaml01cw2_q2_4():
    Xtrn, Xtrn_nm, Ytrn, Xtst_nm, Ytst = myhelper.normalize_data()
    svm = SVC()
    pca = PCA(2)
    pca.fit(Xtrn_nm, Ytrn)
    X_train = pca.transform(Xtrn_nm)
    svm.fit(X_train, Ytrn)
    component1 = X_train[:, 0]
    component2 = X_train[:, 1]
    sigma1 = np.std(component1)
    sigma2 = np.std(component2)
    x1, x2 = np.meshgrid(np.arange(start=int((-5 * sigma1)), stop=int((5 * sigma2)), step=0.01),
                         np.arange(start=int((-5 * sigma1)), stop=int((5 * sigma2)), step=0.01))
    r1, r2 = x1.flatten(), x2.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1, r2))
    plt.contourf(x1, x2, svm.predict(grid).reshape(x1.shape), cmap="coolwarm")
    plt.xlim((-5 * sigma1), (5 * sigma2))
    plt.ylim((-5 * sigma1), (5 * sigma2))
    plt.colorbar(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Class")
    plt.title("Decision Regions spanned by 2D PCA Plane (SVM)")
    plt.xlabel("(-5 * std(PC1)) -> (5 * std(PC2))")
    plt.ylabel("(-5 * std(PC1)) -> (5 * std(PC2))")
    plt.savefig("2_4.png")
    plt.show()
# iaml01cw2_q2_4()


# Q2.5
def iaml01cw2_q2_5():
    Xtrn, Xtrn_nm, Ytrn, Xtst_nm, Ytst = myhelper.normalize_data()
    zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines = [], [], [], [], [], [], [], [], [], []
    classes = [zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines]
    for row in range(Xtrn_nm.shape[0]):
        for cl in range(0, len(classes)):
            if Ytrn[row] == cl:
                classes[cl].append(Xtrn_nm[row])
    Xsmall = []
    for cl in classes:
        for i in range(0, 1000):
            Xsmall.append(cl[i])
    Ysmall = []
    for j in range(0, 10):
        Ysmall.extend([j] * 1000)
    mean_accuracies = []
    c_values = np.logspace(start=-2, stop=3, base=10, num=10)
    for c in c_values:
        svm = SVC(C=c, kernel="rbf", gamma="auto")
        all_accuracies = cross_val_score(estimator=svm, X=Xsmall, y=Ysmall, cv=3)
        mean_accuracies.append(np.mean(all_accuracies))
    max_index = np.where(mean_accuracies == np.max(mean_accuracies))
    optimal_c = c_values[max_index][0]
    print("Highest Mean Accuracy: " + str(np.max(mean_accuracies)))
    print("Optimal C: " + str(optimal_c))
    plt.figure(figsize=(10, 5))
    plt.plot(c_values, mean_accuracies)
    plt.xlabel("C")
    plt.ylabel("Mean CV Classification Accuracy")
    plt.title("Mean Classification Accuracy vs Penalty Parameter C")
    plt.savefig("2_5.png")
    plt.show()
# iaml01cw2_q2_5()


# Q2.6 
def iaml01cw2_q2_6():
    Xtrn, Xtrn_nm, Ytrn, Xtst_nm, Ytst = myhelper.normalize_data()
    zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines = [], [], [], [], [], [], [], [], [], []
    classes = [zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines]
    for row in range(Xtrn_nm.shape[0]):
        for cl in range(0, len(classes)):
            if Ytrn[row] == cl:
                classes[cl].append(Xtrn_nm[row])
    Xsmall = []
    for cl in classes:
        for i in range(0, 1000):
            Xsmall.append(cl[i])
    Ysmall = []
    for j in range(0, 10):
        Ysmall.extend([j] * 1000)
    mean_accuracies = []
    c_values = np.logspace(start=-2, stop=3, base=10, num=10)
    for c in c_values:
        svm = SVC(C=c, kernel="rbf", gamma="auto")
        all_accuracies = cross_val_score(estimator=svm, X=Xsmall, y=Ysmall, cv=3)
        mean_accuracies.append(np.mean(all_accuracies))
    max_index = np.where(mean_accuracies == np.max(mean_accuracies))
    optimal_c = c_values[max_index][0]
    big_svm = SVC(C=optimal_c, kernel="rbf", gamma="auto")
    big_svm.fit(Xtrn_nm, Ytrn)
    train_score = big_svm.score(Xtrn_nm, Ytrn)
    test_score = big_svm.score(Xtst_nm, Ytst)
    print("Training Accuracy = " + str(train_score))
    print("Testing Accuracy = " + str(test_score))
# iaml01cw2_q2_6()

