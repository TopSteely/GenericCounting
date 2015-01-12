from sklearn import linear_model
import os
import numpy as np
import matplotlib.pyplot as plt
import pylab

class_ = 'all'

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    
def main():
    criteria = ['partial', 'threshold', 'complete']
    alphas = [0.000001, 0.001, 0.1, 1, 10, 100]
    epsilons = [0.0005, 0.005, 0.1,  0.5, 1, 5, 10]
    eta0s = [0.0000000001, 0.000000001, 0.00000000001]
    subsample = 15  # 9963 for all
    # get labels, features
    test_imgs, train_imgs = get_seperation()
    X_p, y_p = get_data(class_, test_imgs, train_imgs, subsample, 'training', criteria[0])
    X_p_test, y_p_test = get_data(class_, test_imgs, train_imgs, subsample, 'test', criteria[0])
    print bcolors.WARNING + "partial loaded" + bcolors.ENDC
#    X_t, y_t = get_data(class_, test_imgs, train_imgs, subsample, 'training', criteria[1])
#    X_t_test, y_t_test = get_data(class_, test_imgs, train_imgs, subsample, 'test', criteria[1])
#    print bcolors.WARNING + "threshold loaded" + bcolors.ENDC
#    X_c, y_c = get_data(class_, test_imgs, train_imgs, subsample, 'training', criteria[2])
#    X_c_test, y_c_test = get_data(class_, test_imgs, train_imgs, subsample, 'test', criteria[2])
#    print bcolors.WARNING + "complete loaded" + bcolors.ENDC
    #TODO: save in files
    for c in criteria:
        for a in alphas:
            for es in epsilons:
                for ets in eta0s:
                    if c == 'partial':
                        X = X_p
                        y = y_p
                        X_test = X_p_test
                        y_test = y_p_test
                    elif c == 'threshold':
                        X = X_t
                        y = y_t
                        X_test = X_t_test
                        y_test = y_t_test
                    elif c == 'complete':
                        X = X_c
                        y = y_c
                        X_test = X_c_test
                        y_test = y_c_test

                    # learn
                    clf = linear_model.SGDRegressor(alpha=a, epsilon=es, eta0=ets, fit_intercept=True,
                       l1_ratio=0.15, learning_rate='invscaling', loss='squared_loss',
                       n_iter=200, penalty='l2', power_t=0.25, random_state=None,
                       shuffle=False, verbose=0, warm_start=False)
                    print X.__len__()
                    print y.__len__()
                    clf.fit(X, y)
                    print bcolors.WARNING + "model learned" + bcolors.ENDC
                    # predict
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


def get_seperation():
    file = open('/home/t/Schreibtisch/Thesis/VOCdevkit1/VOC2007/ImageSets/Layout/test.txt')
    test_imgs = []
    train_imgs = []
    for line in file:
        test_imgs.append(int(line))
    for i in range(9963):
        if i not in test_imgs:
            train_imgs.append(i)
    return test_imgs, train_imgs
    
    
def get_data(class_, test_imgs, train_imgs, sample, str, criteria):
    features = []
    labels = []
    class_images = []
    if class_ != 'all':
        # read images with class from file
        file = open('/home/t/Schreibtisch/Thesis/ClassImages/'+ class_+'.txt', 'r')
        for line in file:
            im_nr = int(line)
            if str == 'training' and im_nr in train_imgs:
                class_images.append(im_nr)
            elif str == 'test' and im_nr in test_imgs:
                class_images.append(im_nr)
        print class_images.__len__()
    else:
        if str == 'training':
            class_images = train_imgs
        elif str == 'test':
            class_images = test_imgs
    for i in class_images[0:sample]:
        print i
        fs = get_features(i)
        if fs != []:
            features.extend(fs)
            l = get_labels(i, criteria)
            labels.extend(l)
    return features, labels


def get_features(i):
    features = []
    if os.path.isfile('/home/t/Schreibtisch/Thesis/SS_Boxes/SS_Boxes/'+ (format(i, "06d")) +'.txt'):
        file = open('/home/t/Schreibtisch/Thesis/SS_Boxes/SS_Boxes/'+(format(i, "06d")) +'.txt', 'r')
    else:
        print 'warning /home/t/Schreibtisch/Thesis/SS_Boxes/SS_Boxes/'+ (format(i, "06d")) +'.txt does not exist '
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
    if os.path.isfile('/home/t/Schreibtisch/Thesis/SS_Boxes/Labels/'+(format(i, "06d")) + '_' + criteria + '.txt'):
        file = open('/home/t/Schreibtisch/Thesis/SS_Boxes/Labels/'+(format(i, "06d")) + '_' + criteria + '.txt', 'r')
    else:
        print 'warning /home/t/Schreibtisch/Thesis/SS_Boxes/Labels/'+(format(i, "06d")) + '_' + criteria + '.txt does not exist '
        return labels
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
        if os.path.isfile('/home/t/Schreibtisch/SS_Boxes/SS_Boxes/'+ (format(i, "06d")) +'n.txt'):
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