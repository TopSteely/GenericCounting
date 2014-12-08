from sklearn import linear_model
import random
import os
import numpy as np


def main():
    class_ = 'all'
    subsample = 20


    # get labels, features
    X, y = get_data(class_, subsample)
    X_, y_ = get_negs(subsample)
    X.extend(X_)
    y.extend(y_)
    # learn
    clf = linear_model.SGDRegressor(alpha=0.0001, epsilon=0.1, eta0=0.0000001, fit_intercept=True,
       l1_ratio=0.15, learning_rate='invscaling', loss='squared_loss',
       n_iter=5, penalty='l2', power_t=0.25, random_state=None,
       shuffle=False, verbose=0, warm_start=False)
    clf.fit(X, y)
    # plot


def get_data(class_, sample):
    features = []
    labels = []
    class_images = []
    if class_ != 'all':
        # read images with class from file
        file = open('/home/t/Schreibtisch/Thesis/ClassImages/'+ class_+'.txt', 'r')
        for line in file:
            class_images.append(line)
    else:
        class_images = range(30)
    # shuffle and take only subsample
    random.shuffle(class_images)
    for i in class_images[0:sample]:
        fs = get_features(i)
        l = get_labels(i)
        features.extend(fs)
        labels.extend(l)
    return features, labels


def get_features(i):
    features = []
    if os.path.isfile('/home/t/Schreibtisch/Thesis/Output/'+ (format(i, "06d")) +'.txt'):
        file = open('/home/t/Schreibtisch/Thesis/Output/'+ (format(i, "06d")) +'.txt', 'r')
    else:
        print 'warning'
    for line in file:
        f = []
        tmp = line.split(',')
        for s in tmp:
            f.append(float(s))
        features.append(f)
    return features


def get_labels(i):
    labels = []
    file = open('/home/t/Schreibtisch/Thesis/Labels/'+ (format(i, "06d")) +'.txt', 'r')
    for line in file:
        tmp = line.split()[0]
        labels.append(float(tmp))
    return labels


def get_negs(num):
    features = []
    found = 0
    class_images = range(9963)
    random.shuffle(class_images)
    for i in class_images:
        if os.path.isfile('/home/t/Schreibtisch/Thesis/Output/'+ (format(i, "06d")) +'n.txt'):
            file = open('/home/t/Schreibtisch/Thesis/Output/'+ (format(i, "06d")) +'n.txt', 'r')
            for line in file:
                f = []
                tmp = line.split(',')
                for s in tmp:
                    f.append(float(s))
                features.append(f)
            found += 1
            if found == num:
                return features, np.zeros(features.__len__())



if __name__ == "__main__":
    main()