from sklearn import linear_model
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import pylab

class_ = 'all'


def main():
    criteria = ['partial', 'threshold', 'complete']
    alphas = [0.000001, 0.001, 0.1, 1, 10, 100]
    epsilons = [0.0005, 0.005, 0.1,  0.5, 1, 5, 10]
    eta0s = [0.0000000001, 0.000000001, 0.00000000001]
    # train on 1:8000, test on 8001:end
    for c in criteria:
        for a in alphas:
            for es in epsilons:
                for ets in eta0s:
                    subsample = 2  # 9963 for all

                    # get labels, features
                    X, y = get_data(class_, subsample, 'training', c)
                    #X_, y_ = get_negs(600)

                    # learn
                    clf = linear_model.SGDRegressor(alpha=a, epsilon=es, eta0=ets, fit_intercept=True,
                       l1_ratio=0.15, learning_rate='invscaling', loss='squared_loss',
                       n_iter=200, penalty='l2', power_t=0.25, random_state=None,
                       shuffle=False, verbose=0, warm_start=False)
                    clf.fit(X, y)

                    # predict
                    X_test, y_test = get_data(class_, 1, 'test', c)
                    pred = clf.predict(X_test)
                    mse = sum(((pred - y_test) ** 2)) / y_test.__len__()

                    # plot
                    plt.plot(y_test, pred, 'ro')
                    plt.plot([0,1,2,3,4])
                    plt.axis([0, max(y_test) + 0.01, 0, max(pred) + 0.01])
                    plt.xlabel('true value')
                    plt.ylabel('predicted value')
                    plt.title('Trained on ' + str(X.__len__()) + ' ' + class_ + ' boxes, tested on ' + str(y_test.__len__()) + ' ' + class_ + ' boxes \n alpha=' + str(a) + ', '+\
                              'epsilon=' + str(es) + ' , eta=' + str(ets) + ' - ' + c)
                    plt.text(0.02,0.02,'Mean Squared Error:\n' + str(mse), verticalalignment='bottom',
                                     horizontalalignment='left',
                                     fontsize=10,
                                     bbox={'facecolor':'white', 'alpha':0.6, 'pad':10})
                    #plt.show()
                    pylab.savefig('/home/t/Schreibtisch/Thesis/Plots/' + c + str(a) + str(es) + str(ets) + '.png')


def get_data(class_, sample, str, criteria):
    features = []
    labels = []
    class_images = []
    if class_ != 'all':
        # read images with class from file
        file = open('/home/t/Schreibtisch/Thesis/ClassImages/'+ class_+'.txt', 'r')
        for line in file:
            im_nr = int(line)
            if str == 'training' and im_nr <= 8000:
                class_images.append(im_nr)
            elif str == 'test' and im_nr > 8000:
                class_images.append(im_nr)
        print class_images.__len__()
    else:
        if str == 'training':
            class_images = [216, 243]#range(8000)
        elif str == 'test':
            class_images = [657]#range(8001,9963)
    # shuffle and take only subsample
    # TODO: should i really shuffle here? experiment isn't reproducable
    #random.shuffle(class_images)
    for i in class_images[0:sample]:
        fs = get_features(i)
        if fs != []:
            features.extend(fs)
            l = get_labels(i, criteria)
            labels.extend(l)
    return features, labels


def get_features(i):
    features = []
    if os.path.isfile('/home/t/Schreibtisch/Thesis/Features/'+ (format(i, "06d")) +'.txt'):
        file = open('/home/t/Schreibtisch/Thesis/Features/'+ (format(i, "06d")) +'.txt', 'r')
    else:
        print 'warning'
        return features
    for line in file:
        f = []
        tmp = line.split(',')
        for s in tmp:
            f.append(float(s))
        features.append(f)
    return features


def get_labels(i, criteria):
    labels = []
    file = open('/home/t/Schreibtisch/Thesis/SS_Boxes/Labels/'+ (format(i, "06d")) + '_' + criteria + '.txt', 'r')
    for line in file:
        tmp = line.split()[0]
        labels.append(float(tmp))
    return labels


def get_negs(num):
    features = []
    found = 0
    class_images = range(9963)
    #random.shuffle(class_images)
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