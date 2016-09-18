import os
import time
import random
import numpy as np



def get_image_numbers(test_imgs, train_imgs,class_):
    file = open('/var/scratch/tstahl/IO/ClassImages/'+ class_+'.txt', 'r')
    cl_images = []
    for line in file:
        im_nr = int(line)
        cl_images.append(im_nr)
        
    return filter(lambda x:x in train_imgs,cl_images)
    
def get_seperation():
    file = open('/var/scratch/tstahl/IO/test.txt')
    test_imgs = []
    train_imgs = []
    for line in file:
        test_imgs.append(int(line))
    for i in range(1,9963):
        if i not in test_imgs:
            train_imgs.append(i)
    return test_imgs, train_imgs
    
    
def get_data(class_, test_imgs, train_imgs, start, end, phase, criteria, subsamples):
    features = []
    investigate = []
    labels = []
    if phase == 'training':
        images = train_imgs
    elif phase == 'test':
        images = test_imgs
    for i in images[start:end]:
        fs = get_features(i, subsamples)
#        start = time.time()
#        fs_np = get_features_np(i,subsamples)
#        print time.time() - start
#        raw_input()

# read boxes for coords -> window size

        if os.path.isfile('/var/node436/local/tstahl/Coords_prop_windows/'+ (format(i, "06d")) +'.txt'):
            f = open('/var/node436/local/tstahl/Coords_prop_windows/'+ (format(i, "06d")) +'.txt', 'r')
        else:
            print 'warning'
        boxes = []
        for i_n, line in enumerate(f):
            tmp = line.split(',')
            coord = []
            for s in tmp:
                coord.append(float(s))
            boxes.append(coord)
            if i_n == subsamples - 1:
                break
        if fs != []:
            features.extend(fs)
            l = get_labels(class_,i, criteria, subsamples)
            labels.extend(l)
            for ind in range(len(l)):
                surface_area = (boxes[ind][3] - boxes[ind][1]) * (boxes[ind][2] - boxes[ind][0])
                investigate.append([i, ind, l[ind], surface_area])
        assert (features.__len__() == labels.__len__()), "uneven feature label size!" + str(features.__len__()) + str(labels.__len__())
    return features, labels, investigate
    
    
def get_class_data(class_, test_imgs, train_imgs, start, end, phase, criteria, subsamples):
    features = []
    investigate = []
    labels = []
    class_images = []
    file = open('/var/scratch/tstahl/IO/ClassImages/'+ class_+'.txt', 'r')
    for l in file:
        class_images.append(int(l))
    if phase == 'training':
        images = filter(lambda x:x in train_imgs,class_images)
    elif phase == 'test':
        images = test_imgs
    for i in images[start:end]:
        fs = get_features(i, subsamples)
#        start = time.time()
#        fs_np = get_features_np(i,subsamples)
#        print time.time() - start
#        raw_input()

# read boxes for coords -> window size

        if os.path.isfile('/var/node436/local/tstahl/Coords_prop_windows/'+ (format(i, "06d")) +'.txt'):
            f = open('/var/node436/local/tstahl/Coords_prop_windows/'+ (format(i, "06d")) +'.txt', 'r')
        else:
            print 'warning'
        boxes = []
        for i_n, line in enumerate(f):
            tmp = line.split(',')
            coord = []
            for s in tmp:
                coord.append(float(s))
            boxes.append(coord)
            if i_n == subsamples - 1:
                break
        if fs != []:
            features.extend(fs)
            l = get_labels(class_,i, criteria, subsamples)
            labels.extend(l)
            for ind in range(len(l)):
                surface_area = (boxes[ind][3] - boxes[ind][1]) * (boxes[ind][2] - boxes[ind][0])
                investigate.append([i, ind, l[ind], surface_area])
        assert (features.__len__() == labels.__len__()), "uneven feature label size!" + str(features.__len__()) + str(labels.__len__())
    return features, labels, investigate


def get_features(i, subsamples):
    features = []
    if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt'):
        file = open('/var/node436/local/tstahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt', 'r')
    else:
        print 'warning /var/node436/local/tstahl/Features_prop_windows/SS_Boxes'+ (format(i, "06d")) +'.txt does not exist '
        return features
    for i_n, line in enumerate(file):
        f = []
        tmp = line.split(',')
        for s in tmp:
            f.append(float(s))
        features.append(f)
        if i_n == subsamples - 1:
            break
    return features
    
#def get_features_pd(i, subsamples):
#    features = []
#    if os.path.isfile('/home/stahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt'):
#        features = pd.read_csv('output_list.txt', sep=",", header = None)
#    return features
    
#def get_features_np(i, subsamples):
#    features = []
#    if os.path.isfile('/home/stahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt'):
#        pan = np.loadtxt('/home/stahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt',delimiter=',')        
#    else:
#        print 'warning /home/stahl/Features_prop_windows/SS_Boxes'+ (format(i, "06d")) +'.txt does not exist '
#        return features
#    return pan[0:subsamples]

    
#def get_features_il(i, subsamples):
#    features = []
#    if os.path.isfile('/home/stahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt'):
#        d = iter_loadtxt('/home/stahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt')
#    else:
#        print 'warning /home/stahl/Features_prop_windows/SS_Boxes'+ (format(i, "06d")) +'.txt does not exist '
#        return features
#    return d[0:subsamples]
    

    
#def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
#    def iter_func():
#        with open(filename, 'r') as infile:
#            for _ in range(skiprows):
#                next(infile)
#            for line in infile:
#                line = line.rstrip().split(delimiter)
#                for item in line:
#                    yield dtype(item)
#        iter_loadtxt.rowlength = len(line)
#
#    data = np.fromiter(iter_func(), dtype=dtype)
#    data = data.reshape((-1, iter_loadtxt.rowlength))
#    return data


def get_labels(class_,i, criteria, subsamples):
    labels = []
    if os.path.isfile('/var/node436/local/tstahl/Coords_prop_windows/Labels/Labels/'+(format(i, "06d")) + '_' + class_ + '_' + criteria + '.txt'):
        file = open('/var/node436/local/tstahl/Coords_prop_windows/Labels/Labels/'+(format(i, "06d")) + '_' + class_ + '_' + criteria + '.txt', 'r')
    else:
        print 'warning /var/node436/local/tstahl/Coords_prop_windows/Labels/Labels/'+(format(i, "06d")) + '_' + class_ + '_' + criteria + '.txt does not exist '
        return np.zeros(subsamples)
    for i_l, line in enumerate(file):
        tmp = line.split()[0]
        labels.append(float(tmp))
        if i_l == subsamples - 1:
            break
    return labels