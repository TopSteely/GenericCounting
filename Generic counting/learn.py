from sklearn import linear_model
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpim
import pylab
import pickle
import scipy
import numpy

class_ = 'sheep'
baseline = False

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
                    if os.path.isfile('/home/t/Schreibtisch/Thesis/Models/'+class_+c+str(a)+str(ets)+'_baseline.pickle') and baseline == True:
                        with open('/home/t/Schreibtisch/Thesis/Models/'+class_+c+str(a)+str(ets)+'_baseline.pickle', 'rb') as handle:
                            clf = pickle.load(handle)
                    elif os.path.isfile('/home/t/Schreibtisch/Thesis/Models/'+c+str(a)+str(ets)+'_baseline.pickle') and baseline == True:
                        with open('/home/t/Schreibtisch/Thesis/Models/'+c+str(a)+str(ets)+'_baseline.pickle', 'rb') as handle:
                            clf = pickle.load(handle)
                    elif os.path.isfile('/home/t/Schreibtisch/Thesis/Models/'+class_+c+str(a)+str(ets)+'.pickle') and baseline == False:
                        with open('/home/t/Schreibtisch/Thesis/Models/'+class_+c+str(a)+str(ets)+'.pickle', 'rb') as handle:
                            clf = pickle.load(handle)
                    elif os.path.isfile('/home/t/Schreibtisch/Thesis/Models/'+c+str(a)+str(ets)+'.pickle') and baseline == False:
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
                                n_iter_ = int(np.ceil(10**5 / X_p.__len__()))
                                for epoch in range(n_iter_):
                                    #if epoch % 1000 == 0:
                                    print epoch
                                    clf.partial_fit(X_p, y_p)
                        print "model learned"
                        if baseline == True:
                            with open('/home/t/Schreibtisch/Thesis/Models/'+c+str(a)+str(ets)+'_baseline.pickle', 'wb') as handle:
                                pickle.dump(clf, handle)
                        else:
                            with open('/home/t/Schreibtisch/Thesis/Models/'+c+str(a)+str(ets)+'.pickle', 'wb') as handle:
                                pickle.dump(clf, handle)
                    
                    # predict
                    se = 0.0
                    n = 0
                    max_pred = 0
                    max_y_p_test = 0
                    step = 20
                    ax1 = plt.subplot2grid((3,3), (0,0), colspan=2)
                    ax2 = plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=2)
                    ax3 = plt.subplot2grid((3,3), (1, 2), rowspan=2)
                    pearson_x = []
                    pearson_y = []
                    for minibatch in range(0,200,step):
                        X_p_test, y_p_test, investigate = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + step, 'test', c)
                        if X_p_test != []:
                            if max(y_p_test) > max_y_p_test:
                                max_y_p_test = max(y_p_test)
                            pred = clf.predict(X_p_test)
                            investigate.append(pred)
                            max_over = 0
                            max_under = 0
                            max_over_ind = []
                            max_under_ind = []
                            for row, label in zip(investigate, pred.tolist()):
                                #print row, label
                                if row[2] - label > max_under:
                                    max_under = row[2] - label
                                    max_under_ind = [row[0],row[1]]
                                if label - row[2] > max_over:
                                    max_over = label - row[2]
                                    max_over_ind = [row[0],row[1]]
                            print max_over, max_over_ind, max_under, max_under_ind
                            if max(pred) > max_pred:
                                max_pred = max(pred)
                            se += sum(((pred - y_p_test) ** 2))
                            n += y_p_test.__len__()
                            print (se/n)
                            #if baseline == False:
                                # plot
                            if pearson_x == []:
                                pearson_x = y_p_test
                                pearson_y = pred
                            elif y_p_test.__len__() != 0:
                                pearson_x = numpy.concatenate((pearson_x , y_p_test))
                                pearson_y = numpy.concatenate((pearson_y , pred))
                            y_p_test = numpy.log10(y_p_test)
                            ax2.plot(numpy.log10([0.1,1,2,3,4,5,6,7,8,9,10,11,12,13]), [0.1,1,2,3,4,5,6,7,8,9,10,11,12,13])                                
                            ax2.plot(y_p_test, pred, 'ro')
                            # plot histogram of true value
                            bins = np.arange(-1, 1.5, 0.1)
                            ax1.hist(y_p_test, bins=bins)
                            # plot histogram of predictions
                            hist1, bins1 = np.histogram(pred, bins=50)
                            width = 0.7 * (bins1[1] - bins1[0])
                            center = (bins1[:-1] + bins1[1:]) / 2
                            ax3.bar(center, hist1, align='center', width=width)
                            plt.show()

                    mse = se/n
                    pearson_r = scipy.stats.pearsonr(pearson_x, pearson_y)
                    ax2.axis([-1.1, numpy.log10(max_y_p_test) + 0.01, -1, max_pred + 0.01])
                    ax2.set_xlabel('log(true value)')
                    ax2.set_ylabel('predicted value')
                    ax2.text(-1,0.02,'Mean Squared Error:\n' + str(mse) + '\nPearson:\n' + str(pearson_r[0]), verticalalignment='bottom',
                                     horizontalalignment='left',
                                     fontsize=10,
                                     bbox={'facecolor':'white', 'alpha':0.6, 'pad':10})
                    #plt.show()
                    if baseline == True:
                        pylab.savefig('/home/t/Schreibtisch/Thesis/Plots/' + str(class_) + c + str(a) + str(ets) + '_baseline.png')
                        plt.clf()
                    else:
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
    bla = []
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
            if baseline == True:                    
                features.append(fs)
                tmp = get_labels(i, criteria)
                ll = int(tmp[0])
                labels.append(ll)
                #if phase == 'test':
                #    im = mpim.imread('/home/t/Schreibtisch/Thesis/VOCdevkit1/VOC2007/JPEGImages/'+ (format(i, "06d")) +'.jpg')
                #    plt.imshow(im)
            else:
                features.extend(fs)
                l = get_labels(i, criteria)
                labels.extend(l)
        assert (features.__len__() == labels.__len__()), "uneven feature label size!"
        for ind in range(l.__len__()):
            bla.append([i, ind, l[ind]])
    return features, labels, bla


def get_features(i):
    features = []
    if os.path.isfile('/home/t/Schreibtisch/Thesis/SS_Boxes/SS_Boxes/'+ (format(i, "06d")) +'.txt'):
        file = open('/home/t/Schreibtisch/Thesis/SS_Boxes/SS_Boxes/'+ (format(i, "06d")) +'.txt', 'r')
    else:
        print 'warning /home/t/Schreibtisch/Thesis/SS_Boxes/SS_Boxes/'+ (format(i, "06d")) +'.txt does not exist '
        return features
    if baseline == True:
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