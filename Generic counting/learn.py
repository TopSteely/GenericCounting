from sklearn import linear_model
import os
import numpy as np
import matplotlib.pyplot as plt
import pylab
import pickle

class_ = 'sheep'
debug = False

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
    alphas = [0.001, 0.1, 1, 10, 100]
    #epsilons = [0.0005, 0.005, 0.1,  0.5, 1, 5, 10]
    eta0s = [0.00001, 0.000001, 0.00000001, 0.0000001]
    # get labels, features
    test_imgs, train_imgs = get_seperation()
    for c in criteria:
        for a in alphas:
            #for es in epsilons: # no change of epsilon with loss=squared loss
            for ets in eta0s:
                    print c,a,ets
                    if os.path.isfile('/home/t/Schreibtisch/Thesis/Models/'+class_+c+str(a)+str(ets)+'.pickle'):
                        with open('/home/t/Schreibtisch/Thesis/Models/'+class_+c+str(a)+str(ets)+'.pickle', 'rb') as handle:
                            clf = pickle.load(handle)
                    elif os.path.isfile('/home/t/Schreibtisch/Thesis/Models/'+c+str(a)+str(ets)+'.pickle'):
                        with open('/home/t/Schreibtisch/Thesis/Models/'+c+str(a)+str(ets)+'.pickle', 'rb') as handle:
                            clf = pickle.load(handle)
                    else:
                        # learn
                        clf = linear_model.SGDRegressor(alpha=1, epsilon=0.1, eta0=ets, fit_intercept=True,
                           l1_ratio=0.15, learning_rate='invscaling', loss='squared_loss',
                           n_iter=np.ceil(10**6 / 40), penalty='l2', power_t=0.25, random_state=None,
                           shuffle=True, verbose=1, warm_start=True)
                        #clf = linear_model.LinearRegression()
                        for minibatch in range(0,200,20):
                            X_p, y_p = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + 20, 'training', c)
                            if X_p != []:
                                n_iter_ = int(np.ceil(10**6 / X_p.__len__()))
                                for epoch in range(n_iter_):
                                    #if epoch % 1000 == 0:
                                    print epoch
                                    clf.partial_fit(X_p, y_p)
                        print "model learned"
                        with open('/home/t/Schreibtisch/Thesis/Models/'+c+str(a)+str(ets)+'.pickle', 'wb') as handle:
                            pickle.dump(clf, handle)
                    
                        # predict
                        se = 0.0
                        n = 0
                        max_pred = 0
                        max_y_p_test = 0
                        step = 20
                        for minibatch in range(0,200,step):
                            X_p_test, y_p_test = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + step, 'test', c)
                            if X_p_test != []:
                                if max(y_p_test) > max_y_p_test:
                                    max_y_p_test = max(y_p_test)
                                pred = clf.predict(X_p_test)
                                if max(pred) > max_pred:
                                    max_pred = max(pred)
                                se += sum(((pred - y_p_test) ** 2))
                                n += y_p_test.__len__()
                                if debug == True:
                                    print pred,y_p_test, se, n, (se/n)
                                    raw_input()
                                print (se/n)
                                if debug == False:
                                    # plot
                                    plt.plot(y_p_test, pred, 'ro')
                                    plt.plot([0,1,2,3,4,5,6,7,8,9,10,11,12,13])
                        mse = se/n
                        plt.axis([0, max_y_p_test + 0.01, 0, max_pred + 0.01])
                        plt.xlabel('true value')
                        plt.ylabel('predicted value')
                        plt.title('alpha=' + str(a) + ', '+\
                                  'eta=' + str(ets) + ' - ' + c)
                        plt.text(0.02,0.02,'Mean Squared Error:\n' + str(mse), verticalalignment='bottom',
                                         horizontalalignment='left',
                                         fontsize=10,
                                         bbox={'facecolor':'white', 'alpha':0.6, 'pad':10})
                        #plt.show()
                        pylab.savefig('/home/t/Schreibtisch/Thesis/Plots/' + str(class_) + c + str(a) + str(ets) + '.png')
                        plt.clf()


def get_seperation():
    file = open('/home/t/Schreibtisch/Thesis/VOCdevkit1/VOC2007/ImageSets/Main/test.txt')
    test_imgs = []
    train_imgs = []
    for line in file:
        test_imgs.append(int(line))
    for i in range(9963):
        if i not in test_imgs:
            train_imgs.append(i)
    return test_imgs, train_imgs
    
    
def get_data(class_, test_imgs, train_imgs, start, end, phase, criteria):
    features = []
    labels = []
    class_images = []
    if class_ != 'all':
        # read images with class from file
        file = open('/home/t/Schreibtisch/Thesis/ClassImages/'+ class_+'.txt', 'r')
        for line in file:
            im_nr = int(line)
            if phase == 'training' and im_nr in train_imgs:
                class_images.append(im_nr)
            elif phase == 'test' and im_nr in test_imgs:
                class_images.append(im_nr)
    else:
        if phase == 'training':
            class_images = train_imgs
        elif phase == 'test':
            class_images = test_imgs
    for i in class_images[start:end]:
        print i
        fs = get_features(i)
        if fs != []:
            if debug == True:
                features.append(fs)
                tmp = get_labels(i, criteria)
                ll = int(tmp[0])
                labels.append(ll)
            else:
                features.extend(fs)
                l = get_labels(i, criteria)
                labels.extend(l)
        assert (features.__len__() == labels.__len__()), "uneven feature label size!"
    return features, labels


def get_features(i):
    features = []
    if os.path.isfile('/home/t/Schreibtisch/Thesis/SS_Boxes/SS_Boxes/'+ (format(i, "06d")) +'.txt'):
        file = open('/home/t/Schreibtisch/Thesis/SS_Boxes/SS_Boxes/'+ (format(i, "06d")) +'.txt', 'r')
    else:
        print 'warning /home/t/Schreibtisch/Thesis/SS_Boxes/SS_Boxes/'+ (format(i, "06d")) +'.txt does not exist '
        return features
    if debug == True:
        line = file.readline()
        tmp = line.split(',')
        for s in tmp:
            features.append(float(s))
    else:
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