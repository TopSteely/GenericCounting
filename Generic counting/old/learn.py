from sklearn import linear_model
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpim
import pylab
import pickle
import scipy
import numpy
import math

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
    criteria = ['partial']#, 'threshold0.5', 'threshold0.1', 'threshold0.9']
    alphas = [0.001, 0.1, 1, 10, 100]
    #epsilons = [0.0005, 0.005, 0.1,  0.5, 1, 5, 10]
    eta0s = [0.00001, 0.000001, 0.00000001, 0.0000001]
    # get labels, features
    test_imgs, train_imgs = get_seperation()
    for c in criteria:
        a = alphas[0]
        ets = eta0s[0]
        print c,a,ets
        if os.path.isfile('/home/t/Schreibtisch/Thesis/Models/'+class_+c+str(a)+str(ets)+'_baseline.pickle') and baseline == 1:
            with open('/home/t/Schreibtisch/Thesis/Models/'+class_+c+str(a)+str(ets)+'_baseline.pickle', 'rb') as handle:
                clf = pickle.load(handle)
        elif os.path.isfile('/home/t/Schreibtisch/Thesis/Models/'+c+str(a)+str(ets)+'_baseline.pickle') and baseline == 1:
            with open('/home/t/Schreibtisch/Thesis/Models/'+c+str(a)+str(ets)+'_baseline.pickle', 'rb') as handle:
                clf = pickle.load(handle)
        elif os.path.isfile('/home/t/Schreibtisch/Thesis/Models/'+class_+c+str(a)+str(ets)+'.pickle') and (baseline == 2 or baseline == False):
            with open('/home/t/Schreibtisch/Thesis/Models/'+class_+c+str(a)+str(ets)+'.pickle', 'rb') as handle:
                clf = pickle.load(handle)
        elif os.path.isfile('/home/t/Schreibtisch/Thesis/Models/'+c+str(a)+str(ets)+'.pickle') and (baseline == 2 or baseline == False):
            with open('/home/t/Schreibtisch/Thesis/Models/'+c+str(a)+str(ets)+'.pickle', 'rb') as handle:
                clf = pickle.load(handle)
        else:
            # learn
            clf = linear_model.SGDRegressor(alpha=a, epsilon=0.1, eta0=ets, fit_intercept=True,
               l1_ratio=0.15, learning_rate='invscaling', loss='squared_loss',
               n_iter=np.ceil(10**6 / 40), penalty='l2', power_t=0.25, random_state=None,
               shuffle=True, verbose=1, warm_start=True)
            for minibatch in range(0,200,20):
                X_p, y_p, _ = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + 20, 'training', c)
                if X_p != []:
                    if baseline==1:
                        n_iter_ = int(np.ceil(10**5 / X_p.__len__()))
                    else:
                        n_iter_ = int(np.ceil(10**6 / X_p.__len__()))
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
                    
        clf_sp = linear_model.SGDRegressor(alpha=a, epsilon=0.1, eta0=eta0s[2], fit_intercept=True,
               l1_ratio=0.15, learning_rate='invscaling', loss='squared_loss',
               n_iter=np.ceil(10**6 / 40), penalty='l2', power_t=0.25, random_state=None,
               shuffle=True, verbose=1, warm_start=True)
        samples = 80
        hyper_feats = []
        labels = []
        for minibatch in range(0, 75, 1):
            hyper_feat = []
            X_sp, y_sp, candidates = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + 1, 'training', c)
            img = candidates[0][0]
            target = candidates[0][2]
            if os.path.isfile('/home/t/Schreibtisch/Thesis/SS_Boxes/'+ (format(img, "06d")) +'.txt'):
                f = open('/home/t/Schreibtisch/Thesis/SS_Boxes/'+ (format(img, "06d")) +'.txt', 'r')
            coords = []
            for candidate, line in zip(candidates, f):
                tmp = line.split(',')
                coord = []
                for s in tmp:
                    coord.append(float(s))
                coord.append(candidate[2])
                coords.append(coord)
            # subsample boxes, since we need equal number of features for all images
            shuffle(coords)
#            for box in coords[0:samples]:
#                for box_ in coords[0:samples]:
#                    hyper_feat.append(get_hyper_features(box, box_))
            if coords.__len__() < 80:
                continue
            for box in coords[0:samples]:
                hyper_feat.extend(box)
            assert hyper_feat.__len__() == samples * 5, "not enough feat"
            hyper_feats.append(hyper_feat)
            labels.append(target)
        for epoch in range(int(np.ceil(10**5 / hyper_feats.__len__()))):
            print epoch, hyper_feats.__len__(), labels.__len__()
            clf_sp.partial_fit(hyper_feats, labels)
            
                
        # predict
        se = 0.0
        se_size = 0.0
        se_learned = 0.0
        n = 0
        max_pred = 0
        max_y_p_test = 0
        step = 1
        fig1 = plt.figure(1)
        ax1 = plt.subplot2grid((3,3), (0,0), colspan=2)
        ax2 = plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=2)
        ax3 = plt.subplot2grid((3,3), (1, 2), rowspan=2)
        fig2 = plt.figure(2)
        ax4 = plt.subplot2grid((3,3), (0,0), colspan=2)
        ax5 = plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=2)
        ax6 = plt.subplot2grid((3,3), (1, 2), rowspan=2)
        fig3 = plt.figure(3)
        ax7 = plt.subplot2grid((3,3), (0,0), colspan=2)
        ax8 = plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=2)
        ax9 = plt.subplot2grid((3,3), (1, 2), rowspan=2)
        pearson_x = []
        pearson_y = []
        for minibatch in range(0, 80, step):
            X_p_test, y_p_test, investigate = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + step, 'test', c)
            if X_p_test != []:
                if max(y_p_test) > max_y_p_test:
                    max_y_p_test = max(y_p_test)
                pred = clf.predict(X_p_test)
                img = investigate[0][0]
                print img
                if os.path.isfile('/home/t/Schreibtisch/Thesis/SS_Boxes/'+ (format(img, "06d")) +'.txt'):
                    f = open('/home/t/Schreibtisch/Thesis/SS_Boxes/'+ (format(img, "06d")) +'.txt', 'r')
                img = mpim.imread('/home/t/Schreibtisch/Thesis/VOCdevkit1/VOC2007/JPEGImages/'+ (format(img, "06d")) +'.jpg')
        
                w = img[0].__len__()
                h = img.__len__()
                avgs_size = np.zeros((h, w))
                avgs = np.zeros((h, w))
                boxes = []
                prediction_boxes = []
                hyper_feats = []
                for row, prediction, line in zip(investigate, pred.tolist(), f):
                    tmp = line.split(',')
                    coord = []
                    for s in tmp:
                        coord.append(float(s))
                    #if row[1] == 0:
                        #assert coord[0] == 0 and coord[1] == 0 and coord[2] == h-1 and coord[3] == w-1, "first box not whole image"+str(coord)
                    boxes.append([coord, prediction])
                    coord.append(prediction)
                    prediction_boxes.append(coord)
                if boxes.__len__() < 80:
                    continue
                shuffle(prediction_boxes)
                for box in prediction_boxes[0:samples]:
                    hyper_feats.extend(box)
                image_prediction = clf_sp.predict(hyper_feats)
                
                
                    
                for x in range(w):
                    for y in range(h):
                        pixel_sum = 0
                        pixel_sum_size = 0
                        in_s = 0
                        for b in boxes:
                            in_, p = bool_rect_intersect(b[0], [x,y,x,y])
                            if in_:
                                # average pixel by boxes intersecting with pixel
                                pixel_sum += b[1] / (w * h)
                                # incorporate box size information
                                pixel_sum_size += b[1] * p
                                in_s += 1
                        avgs_size[y,x] = pixel_sum_size / in_s
                        avgs[y,x] = pixel_sum / in_s
                se += sum(((np.sum(avgs) - y_p_test[0]) ** 2))
                se_size += sum(((np.sum(avgs_size) - y_p_test[0]) ** 2))
                se_learned += (image_prediction - y_p_test[0]) ** 2
                n += 1
                print y_p_test[0], np.sum(avgs_size), sum(((np.sum(avgs_size) - y_p_test[0]) ** 2)), '->', (se_size/n)
                print y_p_test[0], np.sum(avgs), sum(((np.sum(avgs) - y_p_test[0]) ** 2)), '->', (se/n)
                print y_p_test[0], image_prediction, (image_prediction - y_p_test[0]) ** 2, '->', (se_learned/n)
                #if baseline == False:
                    # plot
                if pearson_x == []:
                    pearson_x = [y_p_test[0]]
                    pearson_y = [np.sum(avgs)]
                    pearson_y_size = [np.sum(avgs_size)]
                    pearson_y_learned = [image_prediction]
                elif y_p_test.__len__() != 0:
                    pearson_x.append(y_p_test[0])
                    pearson_y.append(np.sum(avgs))
                    pearson_y_size.append(np.sum(avgs_size))
                    pearson_y_learned.append(image_prediction)
                # convert to log-scale for plotting
                y_p_test = numpy.log10(y_p_test)
                plt.figure(1)
                ax2.plot(numpy.log10([0.1,1,2,3,4,5,6,7,8,9,10,11,12,13]), [0.1,1,2,3,4,5,6,7,8,9,10,11,12,13])                                
                ax2.plot(y_p_test[0], np.sum(avgs_size), 'ro')
                # plot histogram of true value
                bins = np.arange(-1, 1.5, 0.1)
                ax1.hist(y_p_test, bins=bins)
                # plot histogram of predictions
                hist1, bins1 = np.histogram(pred, bins=50)
                width = 0.7 * (bins1[1] - bins1[0])
                center = (bins1[:-1] + bins1[1:]) / 2
                ax3.bar(center, hist1, align='center', width=width)
                
                plt.figure(2)
                ax5.plot(numpy.log10([0.1,1,2,3,4,5,6,7,8,9,10,11,12,13]), [0.1,1,2,3,4,5,6,7,8,9,10,11,12,13])                                
                ax5.plot(y_p_test[0], np.sum(avgs), 'ro')
                # plot histogram of true value
                bins = np.arange(-1, 1.5, 0.1)
                ax4.hist(y_p_test, bins=bins)
                # plot histogram of predictions
                hist1, bins1 = np.histogram(pred, bins=50)
                width = 0.7 * (bins1[1] - bins1[0])
                center = (bins1[:-1] + bins1[1:]) / 2
                ax6.bar(center, hist1, align='center', width=width)
                
                plt.figure(3)
                ax8.plot(numpy.log10([0.1,1,2,3,4,5,6,7,8,9,10,11,12,13]), [0.1,1,2,3,4,5,6,7,8,9,10,11,12,13])                                
                ax8.plot(y_p_test[0], image_prediction, 'ro')
                # plot histogram of true value
                bins = np.arange(-1, 1.5, 0.1)
                ax7.hist(y_p_test, bins=bins)
                # plot histogram of predictions
                hist1, bins1 = np.histogram(pred, bins=50)
                width = 0.7 * (bins1[1] - bins1[0])
                center = (bins1[:-1] + bins1[1:]) / 2
                ax9.bar(center, hist1, align='center', width=width)
                
                plt.figure(1)
                plt.show()
                plt.figure(2)
                plt.show()
                plt.figure(3)
                plt.show()
                

        mse = se/n
        mse_size = se_size/n
        mse_learned = se_learned/n
        #print pearson_x, pearson_y
        pearson_r = scipy.stats.pearsonr(pearson_x, pearson_y)
        pearson_r_size = scipy.stats.pearsonr(pearson_x, pearson_y_size)
        pearson_r_learned = scipy.stats.pearsonr(pearson_x, pearson_y_learned)
        plt.figure(1)
        ax2.axis([0 - 0.1, math.log(max(pearson_x)) + 0.1, 0 - 0.1, max(pearson_y_size) + 0.1])
        ax2.set_xlabel('log(true value)')
        ax2.set_ylabel('predicted value')
        ax2.text(-0.1,0.02,'Mean Squared Error:\n' + str(mse_size) + '\nPearson:\n' + str(pearson_r_size[0]), verticalalignment='bottom',
                         horizontalalignment='left',
                         fontsize=10,
                         bbox={'facecolor':'white', 'alpha':0.6, 'pad':10})
        
        
        plt.figure(2)
        ax5.axis([0 - 0.1, math.log(max(pearson_x)) + 0.1, 0 - 0.1, max(pearson_y) + 0.1])
        ax5.set_xlabel('log(true value)')
        ax5.set_ylabel('predicted value')
        ax5.text(-0.1,0.02,'Mean Squared Error:\n' + str(mse) + '\nPearson:\n' + str(pearson_r[0]), verticalalignment='bottom',
                         horizontalalignment='left',
                         fontsize=10,
                         bbox={'facecolor':'white', 'alpha':0.6, 'pad':10})
        plt.figure(3)
        ax8.axis([0 - 0.1, math.log(max(pearson_x)) + 0.1, 0 - 0.1, max(pearson_y_learned) + 0.1])
        ax8.set_xlabel('log(true value)')
        ax8.set_ylabel('predicted value')
        ax8.text(-0.1,0.02,'Mean Squared Error:\n' + str(mse) + '\nPearson:\n' + str(pearson_r_learned[0]), verticalalignment='bottom',
                         horizontalalignment='left',
                         fontsize=10,
                         bbox={'facecolor':'white', 'alpha':0.6, 'pad':10})
        #plt.show()
        if baseline == 1:
            pylab.savefig('/home/t/Schreibtisch/Thesis/Plots/' + str(class_) + c + str(a) + str(ets) + '_baseline_1.png')
            plt.clf()
        elif baseline == 2:
            pylab.savefig('/home/t/Schreibtisch/Thesis/Plots/' + str(class_) + c + str(a) + str(ets) + '_baseline_2.png')
            plt.clf()
        else:
            plt.figure(1)
            #TODO: how set the title in figure not sub
            #plt.title('Baseline 4 with box size information - ' + c)
            pylab.savefig('/home/t/Schreibtisch/Thesis/Plots/' + str(class_) + c + str(a) + str(ets) + '_baseline4_size.png')
            plt.figure(2)
            #plt.title('Baseline 4 - ' + c)
            pylab.savefig('/home/t/Schreibtisch/Thesis/Plots/' + str(class_) + c + str(a) + str(ets) + '_baseline4_wo_size.png')
            #plt.clf()


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
    investigate = []
    training_images = []
    class_images = []
    trained = False
    if class_ != 'all':
        # read images with class from file
        file = open('/home/t/Schreibtisch/Thesis/ClassImages/'+ class_+'.txt', 'r')
        for line in file:
            im_nr = int(line)
            class_images.append(int(line))
            if phase == 'training' and im_nr in train_imgs:
                training_images.append(im_nr)
            elif phase == 'test' and im_nr in test_imgs:
                training_images.append(im_nr)
    else:
        if phase == 'training':
            training_images = train_imgs
        elif phase == 'test':
            training_images = test_imgs
    for i in training_images[start:end]:
        print i
        fs = get_features(i)
        if fs != []:
            trained = True
            if baseline == 1 or baseline == 2:                    
                features.append(fs)
                tmp = get_labels(i, criteria)
                ll = int(tmp[0])
                labels.append(ll)
                investigate.append([i, 0, ll])
                #if phase == 'test':
                #    im = mpim.imread('/home/t/Schreibtisch/Thesis/VOCdevkit1/VOC2007/JPEGImages/'+ (format(i, "06d")) +'.jpg')
                #    plt.imshow(im)
            else:
                features.extend(fs)
                l = get_labels(i, criteria)
                labels.extend(l)
                for ind in range(l.__len__()):
                    investigate.append([i, ind, l[ind]])
        assert (features.__len__() == labels.__len__()), "uneven feature label size!" + str(features.__len__()) + str(labels.__len__())
    # if baseline 1 training, add some non-class images to have zero object possibility as well
    if baseline == 1 and phase == 'training' and trained == True:
        for i in range(start + 1, end + 1):
            if i not in class_images:
                fs = get_features(i)
                if fs != []:
                    features.append(fs)
                    labels.append(0)
    return features, labels, investigate


def get_features(i):
    features = []
    if os.path.isfile('/home/t/Schreibtisch/Thesis/SS_Boxes/SS_Boxes/'+ (format(i, "06d")) +'.txt'):
        file = open('/home/t/Schreibtisch/Thesis/SS_Boxes/SS_Boxes/'+ (format(i, "06d")) +'.txt', 'r')
    else:
        print 'warning /home/t/Schreibtisch/Thesis/SS_Boxes/SS_Boxes/'+ (format(i, "06d")) +'.txt does not exist '
        return features
    if baseline == 1 or baseline == 2:
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
    if os.path.isfile('/home/t/Schreibtisch/Thesis/SS_Boxes/Labels/'+(format(i, "06d")) + '_' + class_ + '_' + criteria + '.txt'):
        file = open('/home/t/Schreibtisch/Thesis/SS_Boxes/Labels/'+(format(i, "06d")) + '_' + class_ + '_' + criteria + '.txt', 'r')
    else:
        print 'warning /home/t/Schreibtisch/Thesis/SS_Boxes/Labels/'+(format(i, "06d")) + '_' + class_ + '_' + criteria + '.txt does not exist '
        return labels
    for line in file:
        tmp = line.split()[0]
        labels.append(float(tmp))
    return labels


def get_negs(num):
    features = []
    found = 0
    training_images = range(9963)
    #random.shuffle(training_images)
    for i in training_images:
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


def bool_rect_intersect(A, B):
    return not (B[0]>A[2] or B[2]<A[0] or B[3]<A[1] or B[1]>A[3]), 1/((A[2]- A[0] + 1)*(A[3]-A[1] + 1))
        #return !(r2.left > r1.right || r2.right < r1.left || r2.top > r1.bottom ||r2.bottom < r1.top);

def get_hyper_features(A, B):
    in_, _ = bool_rect_intersect(A, B)
    if not in_:
        return 0, 0, 0
    else:
        left = max(A[0], B[0]);
        right = min(A[2], B[2]);
        top = max(A[1], B[1]);
        bottom = min(A[3], B[3]);
        intersection = [left, top, right, bottom];
        surface_intersection = (intersection[2]-intersection[0])*(intersection[3]-intersection[1]);
        surface_A = (A[2]- A[0])*(A[3]-A[1]);
        surface_B = (B[2]- B[0])*(B[3]-B[1]);
        surface_union = surface_A + surface_B - surface_intersection
        return surface_intersection / surface_union, surface_intersection / surface_A, surface_intersection / surface_B
        
    
    
if __name__ == "__main__":
    main()