
import numpy as np
from helpers import iaml01cw2_helpers as helper
import os

# s1817812
# Question 1:


def normalize_data():
    Xtrn, Ytrn, Xtst, Ytst = helper.load_FashionMNIST(os.getcwd())
    Xtrn_orig = Xtrn.copy()
    Xtst_orig = Xtst.copy()
    Xtrn = Xtrn / 255.0
    Xtst = Xtst / 255.0
    Xmean = np.mean(Xtrn, axis=0)
    # instantiate Xtrn_nm, Xtst_nm
    Xtrn_nm = np.ones(Xtrn.shape)
    Xtst_nm = np.ones(Xtst.shape)
    # fill Xtrn_nm, Xtst_nm
    for row in range(Xtrn.shape[0] - 1):
        Xtrn_nm[row] = Xtrn[row] - Xmean
    for row in range(Xtst.shape[0] - 1):
        Xtst_nm[row] = Xtst[row] - Xmean
    return np.array([Xtrn, Xtrn_nm, Ytrn, Xtst_nm, Ytst])


def get_mean_img(c):
    means = np.mean(c, axis=0)
    image = means.reshape((28, 28))
    return image


def get_differences(c):
    diffs = []
    for i in range(len(c)):
        img = c[i].reshape((28, 28))
        diffs.append(abs(img - get_mean_img(c)))
    return diffs


def get_total_differences(c):
    images = get_differences(c)
    sums = []
    for i in range(len(images)):
        sums.append(np.sum(images[i]))
    return sums


def get_closest_image(c):
    differences = get_total_differences(c)
    loc = np.where(differences == np.min(differences))
    loc = int(loc[0])
    closest_img = c[loc].reshape((28, 28))
    return closest_img


def get_second_closest_image(c):
    differences = get_total_differences(c)
    new_diffs = np.delete(differences, np.min(differences))
    loc = np.where(new_diffs == np.min(new_diffs))
    loc = int(loc[0])
    second_closest_img = c[loc].reshape((28, 28))
    return second_closest_img


def get_furthest_image(c):
    differences = get_total_differences(c)
    loc = np.where(differences == np.max(differences))
    loc = int(loc[0])
    furthest_img = c[loc].reshape((28, 28))
    return furthest_img


def get_second_furthest_image(c):
    differences = get_total_differences(c)
    new_diffs = np.delete(differences, np.max(differences))
    loc = np.where(new_diffs == np.max(new_diffs))
    loc = int(loc[0])
    second_furthest_img = c[loc].reshape((28, 28))
    return second_furthest_img

