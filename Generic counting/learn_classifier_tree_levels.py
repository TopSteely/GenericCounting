# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:20:18 2015

@author: root
"""

from sklearn import linear_model, preprocessing
import matplotlib
matplotlib.use('agg')
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy
import math
import sys
import random
import pylab as pl
import networkx as nx
import pyximport; pyximport.install(pyimport = True)
import get_overlap_ratio
import itertools
from get_intersection import get_intersection
from collections import deque
from itertools import chain, islice
from get_intersection_count import get_intersection_count
#from count_per_lvl import iep,sums_of_all_cliques,count_per_level
import matplotlib.colors as colors
from load import get_seperation, get_data,get_image_numbers,get_class_data
import matplotlib.image as mpimg
from utils import create_tree, find_children, sort_boxes, surface_area, extract_coords, get_set_intersection
from ml import tree_level_regression, tree_level_loss, count_per_level, sums_of_all_cliques
import time
import matplotlib.cm as cmx
from scipy import optimize
import time
import cProfile
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from matplotlib.patches import Rectangle
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier

#class_ = 'sheep'
baseline = False
add_window_size = False
iterations = 1000
subsampling = False
c = 'partial'
normalize = True
prune = False
delta = math.pow(10,-3)
features_used = 5
less_features = False
learn_intersections = True
squared_hinge_loss = False
prune_fully_covered = True
prune_tree_levels = 2
jans_idea = True


def minibatch_(functions, clf,scaler,w, loss__,mse,hinge1,hinge2,full_image,alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance, mode):
    if mode == 'loss_test' or mode == 'loss_scikit_test' or mode == 'levels_test' or mode == 'extract_test':
        X_p, y_p, inv = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + 1, 'test', c,subsamples)                
    elif mode == 'train':
        X_p, y_p, inv = get_class_data(class_, test_imgs, train_imgs, minibatch, minibatch + 1, 'training', c,subsamples)        
    else:
         X_p, y_p, inv = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + 1, 'training', c,subsamples)    
    if X_p != []:
        boxes = []
        ground_truth = inv[0][2]
        img_nr = inv[0][0]
        if less_features:
            X_p = [fts[0:features_used] for fts in X_p]
        if os.path.isfile('/var/node436/local/tstahl/Coords_prop_windows/'+ (format(img_nr, "06d")) +'.txt'):
            f = open('/var/node436/local/tstahl/Coords_prop_windows/'+ (format(img_nr, "06d")) +'.txt', 'r')
        else:
            print 'warning'
        for line, y in zip(f, inv):
            tmp = line.split(',')
            coord = []
            for s in tmp:
                coord.append(float(s))
            boxes.append([coord, y[2]])
        #assert(len(boxes)<500)
        boxes, y_p, X_p = sort_boxes(boxes, y_p, X_p, 0,5000)
        
        if os.path.isfile('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d"))):
            gr = open('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d")), 'r')
        else:
            gr = []
        ground_truths = []
        for line in gr:
           tmp = line.split(',')
           ground_truth = []
           for s in tmp:
              ground_truth.append(int(s))
           ground_truths.append(ground_truth)
        
        #prune boxes
        pruned_x = []
        pruned_y = []
        pruned_boxes = []
        if prune:
            for i, y_ in enumerate(y_p):
                if y_ > 0:
                    pruned_x.append(X_p[i])
                    pruned_y.append(y_p[i])
                    pruned_boxes.append(boxes[i])
        else:
            pruned_x = X_p
            pruned_y = y_p
            pruned_boxes = boxes
        
        if subsampling and pruned_boxes > subsamples:
            pruned_x = pruned_x[0:subsamples]
            pruned_y = pruned_y[0:subsamples]
            pruned_boxes = pruned_boxes[0:subsamples]
            
            
        # create_tree
        G, levels = create_tree(pruned_boxes)
        
        #prune tree to only have levels which fully cover the image, tested
        if prune_fully_covered:
            nr_levels_covered = 100
            total_size = surface_area(pruned_boxes, levels[0])
            for level in levels:
                sa = surface_area(pruned_boxes, levels[level])
                sa_co = sa/total_size
                if sa_co != 1.0:
                    G.remove_nodes_from(levels[level])
                else:
                    nr_levels_covered = level
            levels = {k: levels[k] for k in range(0,nr_levels_covered + 1)}
            
        # prune levels, speedup + performance 
        levels = {k:v for k,v in levels.iteritems() if k<prune_tree_levels}
        
        coords = []
        features = []
        f_c = []
        f = []
        
        #either subsampling or prune_fully_covered
        #assert(subsampling != prune_fully_covered)
        
        if subsampling:
            if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/upper_levels/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples)):
                f_c = open('/var/node436/local/tstahl/Features_prop_windows/upper_levels/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples), 'r+')
            else:
                if mode == 'extract_train' or mode == 'extract_test':                
                    print 'coords for %s with %s samples have to be extracted'%(img_nr,subsamples)
                    f_c = open('/var/node436/local/tstahl/Features_prop_windows/upper_levels/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples), 'w')
                    for level in levels:
                        levl_boxes = extract_coords(levels[level], pruned_boxes)
                        if levl_boxes != []:
                            for lvl_box in levl_boxes:
                                if lvl_box not in coords:
                                    coords.append(lvl_box)
                                    f_c.write('%s,%s,%s,%s'%(lvl_box[0],lvl_box[1],lvl_box[2],lvl_box[3]))
                                    f_c.write('\n')
                    f_c.close()
                    print 'features for %s with %s samples have to be extracted'%(img_nr,subsamples)
                    os.system('export PATH=$PATH:/home/koelma/impala/lib/x86_64-linux-gcc')
                    os.system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/koelma/impala/third.13.03/x86_64-linux/lib')
                    #print "EuVisual /var/node436/local/tstahl/Images/%s.jpg /var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep_%s_%s.txt --eudata /home/koelma/EuDataBig --imageroifile /var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_%s.txt"%((format(img_nr, "06d")),format(img_nr, "06d"),subsamples,format(img_nr, "06d"),subsamples)
                    os.system("EuVisual /var/node436/local/tstahl/Images/%s.jpg /var/node436/local/tstahl/Features_prop_windows/Features_upper/%s_%s_%s.txt --eudata /home/koelma/EuDataBig --imageroifile /var/node436/local/tstahl/Features_prop_windows/upper_levels/%s_%s_%s.txt"%(class_,(format(img_nr, "06d")),format(img_nr, "06d"),subsamples,class_,format(img_nr, "06d"),subsamples))
                    if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/upper_levels/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples)):
                        f_c = open('/var/node436/local/tstahl/Features_prop_windows/upper_levels/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples), 'r')
                    else:
                        f_c = []
            coords = []
                
            if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/Features_upper/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples)):
                f = open('/var/node436/local/tstahl/Features_prop_windows/Features_upper/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples), 'r') 
                
                
        elif prune_fully_covered:
            if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d"))):
                f_c = open('/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d")), 'r+')
                
                
            else:
                if mode == 'extract_train' or mode == 'extract_test':                
                    print 'coords for %s with fully_cover_tree samples have to be extracted'%(img_nr)
                    f_c = open('/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d")), 'w')
                    for level in levels:
                        levl_boxes = extract_coords(levels[level], pruned_boxes)
                        if levl_boxes != []:
                            for lvl_box in levl_boxes:
                                if lvl_box not in coords:
                                    coords.append(lvl_box)
                                    f_c.write('%s,%s,%s,%s'%(lvl_box[0],lvl_box[1],lvl_box[2],lvl_box[3]))
                                    f_c.write('\n')
                    f_c.close()
                    print 'features for %s with fully_cover_tree samples have to be extracted'%(img_nr)
                    os.system('export PATH=$PATH:/home/koelma/impala/lib/x86_64-linux-gcc')
                    os.system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/koelma/impala/third.13.03/x86_64-linux/lib')
                    #print "EuVisual /var/node436/local/tstahl/Images/%s.jpg /var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep_%s_%s.txt --eudata /home/koelma/EuDataBig --imageroifile /var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_%s.txt"%((format(img_nr, "06d")),format(img_nr, "06d"),subsamples,format(img_nr, "06d"),subsamples)
                    print "EuVisual /var/node436/local/tstahl/Images/%s.jpg /var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep_%s_fully_cover_tree.txt --eudata /home/koelma/EuDataBig --imageroifile /var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt"%((format(img_nr, "06d")),format(img_nr, "06d"),format(img_nr, "06d"))
                    os.system("EuVisual /var/node436/local/tstahl/Images/%s.jpg /var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep_%s_fully_cover_tree.txt --eudata /home/koelma/EuDataBig --imageroifile /var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt"%((format(img_nr, "06d")),format(img_nr, "06d"),format(img_nr, "06d")))
                    if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d"))):
                        f_c = open('/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d")), 'r')
                    else:
                        f_c = []
            coords = []
                
            if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d"))):
                f = open('/var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d")), 'r') 
                        
                
        else:
            if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep%s.txt'%(format(img_nr, "06d"))):
                f = open('/var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep%s.txt'%(format(img_nr, "06d")), 'r') 
            if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep%s.txt'%(format(img_nr, "06d"))):
                f_c = open('/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep%s.txt'%(format(img_nr, "06d")), 'r+')
                
        if f_c != []:
            for i,line in enumerate(f_c):
                str_ = line.rstrip('\n').split(',')
                cc = []
                for s in str_:
                   cc.append(float(s))
                coords.append(cc)
        if f != []:
            for i,line in enumerate(f):
                str_ = line.rstrip('\n').split(',')  
                ff = []
                for s in str_:
                   ff.append(float(s))
                features.append(ff)
        #assert len(coords) == len(features)
        
        # append x,y of intersections
        if learn_intersections:
            for inters,coord in zip(features,coords):
#                if inters not in pruned_x:
                pruned_x.append(inters)
                ol = 0.0
                ol = get_intersection_count(coord, ground_truths)
                pruned_y.append(ol)
                
        if mode == 'mean_variance':
            print 'normalizing'
            sum_x += np.array(pruned_x).sum(axis=0)
            n_samples += len(pruned_x)
            sum_sq_x +=  (np.array(pruned_x)**2).sum(axis=0)
            scaler.partial_fit(pruned_x)  # Don't cheat - fit only on training data
            return sum_x,n_samples,sum_sq_x, scaler
            
        if less_features:
            features = [fts[0:features_used] for fts in features]
        #normalize
        norm_x = []
        if normalize and (mode != 'extract_train' and mode != 'extract_test'):
#            for p_x in pruned_x:
#                norm_x.append((p_x-mean)/variance)
            norm_x = scaler.transform(pruned_x)
            if features != []:
                features = scaler.transform(features)
        else:
            norm_x = pruned_x
        data = (G, levels, pruned_y, norm_x, pruned_boxes, ground_truths, alphas)
        sucs = nx.dfs_successors(G)
        
        predecs = nx.dfs_predecessors(G)
        
        #preprocess: node - children
        children = {}
        last = -1
        for node,children_ in zip(sucs.keys(),sucs.values()):
            if node != last+1:
                for i in range(last+1,node):
                    children[i] = []
                children[node] = children_
            elif node == last +1:
                children[node] = children_
            last = node
        if mode == 'train':
            if alphas[0] == 0: #if we don't learn the proposals, we learn just the levels: better, because every level has same importance and faster
                for level in levels:
                    print 'level' , level
                    if img_nr in functions:
                        if level in functions[img_nr]:
                            function = functions[img_nr][level]
                        else:
                            function = []
                    else:
                        functions[img_nr] = {}
                        function = []
                    w, function = tree_level_regression(class_,function,levels,level,features,coords,scaler,w,norm_x,pruned_y,None,predecs,children,pruned_boxes,learning_rate,alphas,img_nr,jans_idea)
                    if level not in functions[img_nr]:
                        functions[img_nr][level] = function
                return w, len(pruned_y), len(levels)
            else: #if we learn proposals, levels with more proposals have more significance...., slow - need to change
                nodes = list(G.nodes())
                for node in nodes:
                    for num,n in enumerate(levels.values()):
                        if node in n:
                            level = num
                            break
                    if img_nr in functions:
                        if level in functions[img_nr]:
                            function = functions[img_nr][level]
                        else:
                            function = []
                    else:
                        functions[img_nr] = {}
                        function = []
                    w, function = tree_level_regression(class_,function,levels,level,features,coords,scaler,w,norm_x,pruned_y,node,predecs,children,pruned_boxes,learning_rate,alphas,img_nr)
                    #TODO: train regressor/classifier that predicts/chooses level. Features: level, number of proposals, number of intersections, avg size of proposal, predictions(for regressor), etc.
                    if level not in functions[img_nr]:
                        functions[img_nr][level] = function
            return w, len(pruned_y), len(G.nodes())
        elif mode == 'scikit_train':
            clf.partial_fit(norm_x,pruned_y)
            return clf
        elif mode == 'loss_train':
            if clf.predict(np.array(X_p[0]).reshape(1, -1)) == 0 and pruned_y[0] == 0:
                loss__.append(0)
            elif clf.predict(np.array(X_p[0]).reshape(1, -1)) == 0 and pruned_y[0] > 0:
                loss__.append(pruned_y[0]**2)
            else:
                loss__.append(tree_level_loss(class_,features,coords,scaler, w, data, predecs, children,img_nr,-1,functions))
            return loss__
        elif mode == 'loss_test':
            if clf.predict(np.array(X_p[0]).reshape(1, -1)) == 0 and pruned_y[0] == 0:
                loss__.append(0)
                full_image.append([pruned_y[0],0])
            elif clf.predict(np.array(X_p[0]).reshape(1, -1)) == 0 and pruned_y[0] > 0:
                loss__.append(pruned_y[0]**2)
                full_image.append([pruned_y[0],0])
            else:
                loss__.append(tree_level_loss(class_,features,coords,scaler, w, data, predecs, children,img_nr,-1,functions))
                cpl = max(0, np.dot(w,np.array(norm_x[0]).T))
                full_image.append([pruned_y[0],cpl])
            return loss__,full_image
        elif mode == 'loss_scikit_test' or mode == 'loss_scikit_train':
            loss__.append(((clf.predict(norm_x) - pruned_y)**2).sum())
            return loss__ 
        elif mode == 'levels_train' or mode == 'levels_test':
            #im = mpimg.imread('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg')
            if clf.predict(np.array(X_p[0]).reshape(1, -1)) == 0 and pruned_y[0] == 0:
                cpls = []
                truelvls = []
                for level in levels:
                    cpls.append(0)
                    truelvls.append(0)
            elif clf.predict(np.array(X_p[0]).reshape(1, -1)) == 0 and pruned_y[0] > 0:
                loss__.append(pruned_y[0]**2)
                cpls = []
                truelvls = []
                for level in levels:
                    cpls.append(0)
                    truelvls.append(pruned_y[0])
            else:
                preds = []
                for i,x_ in enumerate(norm_x):
                    preds.append(np.dot(w, x_))
                cpls = []
                truelvls = []
                used_boxes_ = []
                total_size = surface_area(pruned_boxes, levels[0])
                fully_covered_score = 0.0
                fully_covered_score_lvls = 0.0
                covered_levels = []
                print mode, len(levels)
                for level in levels:
                    function = functions[img_nr][level]
                    cpl,used_boxes,_ = count_per_level([],class_,features,coords,scaler,w, preds, img_nr, pruned_boxes,levels[level], '',function)
                    # clipp negative predictions
                    cpl = max(0,cpl)
                    if used_boxes != []:
                        used_boxes_.append(used_boxes[0][1])
                    tru = y_p[0]
                    cpls.append(cpl)
                    sa = surface_area(pruned_boxes, levels[level])
                    sa_co = sa/total_size
                    if sa_co == 1.0:
                       fully_covered_score += cpl
                       fully_covered_score_lvls += 1
                       covered_levels.append(cpl)
                    truelvls.append(tru)
            return cpls,truelvls
        
            
def main():
    test_imgs, train_imgs = get_seperation()
    # learn
#    if os.path.isfile('/var/node436/local/tstahl/Models/'+class_+c+'normalized_constrained.pickle'):
#        with open('/var/node436/local/tstahl/Models/'+class_+c+'normalized_constrained.pickle', 'rb') as handle:
#            w = pickle.load(handle)
#    else:
    gamma = 0.1
    #subsamples_ = [5,8,12]
    if subsampling:
        subsamples = 5
    else:
        subsamples = 100000
    learning_rates = [math.pow(10,-4)]
    learning_rates_ = {}
    if less_features:
        weights_sample = random.sample(range(features_used), 2)
    else:
        weights_sample = random.sample(range(4096), 10)
    all_alphas = [1]
    regs = [1e-5]
    n_samples = 0.0
    if less_features:
        sum_x = np.zeros(features_used)
        sum_sq_x = np.zeros(features_used)
    else:
        sum_x = np.zeros(4096)
        sum_sq_x = np.zeros(4096)
    if len(sys.argv) != 2:
        print 'wrong arguments'
        exit()
    mous = 'whole'
    global class_
    class_ = sys.argv[1]
    
    print 'learning', class_
    if mous != 'whole':
        train_imgs = get_image_numbers(test_imgs,train_imgs,class_)
    plt.figure()
    mean = []
    variance = []
    scaler = []
    functions = {}
#    train_imgs = train_imgs[0:37]
#    test_imgs = test_imgs[0:37]
    class_images_ = get_image_numbers(test_imgs,train_imgs,class_)
    training_class_images = filter(lambda x:x in train_imgs,class_images_)
    shuffled = range(0,len(training_class_images))
    converged = 10000000
    plot_loss = []
     
    print 'extract features'
#        for minibatch in range(0,len(train_imgs)):
#            minibatch_(None,None,scaler,[], [],[],[],[],[],[],[],test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,None,None,'extract_train')
#            minibatch_(None,None,scaler,[], [],[],[],[],[],[],[],test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,None,None,'extract_test')
 
    #normalize
    learning_rate0 = learning_rates[0]
    alpha1 =         all_alphas[0]
    reg = regs[0]
    alphas = [1-alpha1,alpha1,reg]
    
    clf1 = SVC(verbose=True)
    clf2 = LinearSVC(verbose=True, max_iter=10000)
    clfs = []
    false_n = []
    false_p = []
    true_p = []
    true_n = []
    false_n_t = []
    false_p_t = []
    true_p_t = []
    true_n_t = []
    false_neg_SVC = 0
    false_pos_SVC = 0
    true_pos_SVC = 0
    true_neg_SVC = 0
    false_neg_t_SVC = 0
    false_pos_t_SVC = 0
    true_pos_t_SVC = 0
    true_neg_t_SVC = 0
    false_negLinearSVC = 0
    false_posLinearSVC = 0
    true_posLinearSVC = 0
    true_negLinearSVC = 0
    false_neg_tLinearSVC = 0
    false_pos_tLinearSVC = 0
    true_pos_tLinearSVC = 0
    true_neg_tLinearSVC = 0
    X_ = []
    y_ = []
    X_test = []
    y_test = []
    for minibatch in range(0,len(train_imgs)):
        X_p, y_p, inv = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + 1, 'training', c,1)
        X_.append(X_p[0])
        if y_p[0] > 0:
            y_.append(1)
        else:
            y_.append(0)
    for minibatch in range(0,len(test_imgs)):
            X_temp, y_temp, _ = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + 1, 'test', c,1)
            X_test.append(X_temp)
            y_test.append(y_temp)
    for iters in np.arange(1,30):
        # learn classifier, SVC, LinearSVC
        clf = SGDClassifier(verbose=True, n_iter=iters)
        clf.fit(X_,y_)
        if iters == 1:
            clf1.fit(X_,y_)
            clf2.fit(X_,y_)
        false_neg = 0
        false_pos = 0
        true_pos = 0
        true_neg = 0
        false_neg_t = 0
        false_pos_t = 0
        true_pos_t = 0
        true_neg_t = 0
        for minibatch in range(0,len(train_imgs)):
            X_p = X_[minibatch]
            y_p = y_[minibatch]
            p = clf.predict(np.array(X_p).reshape(1, -1))
            if p[0] == 1 and y_p >= 1:
                true_pos_t += 1
            if p[0] == 1 and y_p == 0:
                false_pos_t += 1
            if p[0] == 0 and y_p == 0:
                true_neg_t += 1
            if p[0] == 0 and y_p >= 1:
                false_neg_t += 1
            if iters == 1:
                p1 = clf1.predict(np.array(X_p).reshape(1, -1))
                p2 = clf2.predict(np.array(X_p).reshape(1, -1))
                if y_p >= 1:
                    if p1[0] == 1:
                        true_pos_t_SVC += 1
                    if p2[0] == 1:
                        true_pos_tLinearSVC += 1
                    if p1[0] == 0:
                        false_neg_t_SVC += 1
                    if p2[0] == 0:
                        false_neg_tLinearSVC += 1
                else:
                    if p1[0] == 1:
                        false_pos_t_SVC += 1
                    if p2[0] == 1:
                        false_pos_tLinearSVC += 1
                    if p1[0] == 0:
                        true_neg_t_SVC += 1
                    if p2[0] == 0:
                        true_neg_tLinearSVC += 1
        for minibatch in range(0,len(test_imgs)):
            X_p = X_test[minibatch]
            y_p = y_test[minibatch]
            p = clf.predict(np.array(X_p).reshape(1, -1))
            if p[0] == 1 and y_p[0] >= 1:
                true_pos += 1
            if p[0] == 1 and y_p[0] == 0:
                false_pos += 1
            if p[0] == 0 and y_p[0] == 0:
                true_neg += 1
            if p[0] == 0 and y_p[0] >= 1:
                false_neg += 1
            if iters == 1:
                p1 = clf1.predict(np.array(X_p).reshape(1, -1))
                p2 = clf2.predict(np.array(X_p).reshape(1, -1))
                if y_p[0] >= 1:
                    if p1[0] == 1:
                        true_pos_SVC += 1
                    if p2[0] == 1:
                        true_posLinearSVC += 1
                    if p1[0] == 0:
                        false_neg_SVC += 1
                    if p2[0] == 0:
                        false_negLinearSVC += 1
                else:
                    if p1[0] == 1:
                        false_pos_SVC += 1
                    if p2[0] == 1:
                        false_posLinearSVC += 1
                    if p1[0] == 0:
                        true_neg_SVC += 1
                    if p2[0] == 0:
                        true_negLinearSVC += 1
        clfs.append(clf)
        false_n.append(false_neg)
        false_p.append(false_pos)
        true_p.append(true_pos)
        true_n.append(true_neg)
        false_n_t.append(false_neg_t)
        false_p_t.append(false_pos_t)
        true_p_t.append(true_pos_t)
        true_n_t.append(true_neg_t)
    fig,ax = plt.subplots(2,2,sharex=True)
    svc1, = ax[0,0].plot(np.arange(1,30),np.ones(29) * true_pos_SVC,'-ro',label='SVC')
    svc2, = ax[0,0].plot(np.arange(1,30),np.ones(29) * true_pos_t_SVC,'-rx',label='SVC train')
    lin1, = ax[0,0].plot(np.arange(1,30),np.ones(29) * true_posLinearSVC,'-go',label='LinearSVC')
    lin2, = ax[0,0].plot(np.arange(1,30),np.ones(29) * true_pos_tLinearSVC,'-gx',label='LinearSVC train')
    sgd1, = ax[0,0].plot(true_p,'-bo',label='SGDClassifier')
    sgd2, = ax[0,0].plot(true_p_t,'-bx',label='SGDClassifier train')
    ax[0, 0].set_title('True Positives')
    
    svc1, = ax[0,1].plot(np.arange(1,30),np.ones(29) * true_neg_SVC,'-ro',label='SVC')
    svc2, =ax[0,1].plot(np.arange(1,30),np.ones(29) * true_neg_t_SVC,'-rx',label='SVC train')
    lin1, = ax[0,1].plot(np.arange(1,30),np.ones(29) * true_negLinearSVC,'-go',label='LinearSVC')
    lin2, = ax[0,1].plot(np.arange(1,30),np.ones(29) * true_neg_tLinearSVC,'-gx',label='LinearSVC train')
    sgd1, = ax[0,1].plot(true_n,'-bo',label='SGDClassifier')
    sgd2, = ax[0,1].plot(true_n_t,'-bx',label='SGDClassifier train')
    ax[0, 1].set_title('True Negatives')
    
    svc1, = ax[1,1].plot(np.arange(1,30),np.ones(29) * false_neg_SVC,'-ro',label='SVC')
    svc2, =ax[1,1].plot(np.arange(1,30),np.ones(29) * false_neg_t_SVC,'-rx',label='SVC train')
    lin1, = ax[1,1].plot(np.arange(1,30),np.ones(29) * false_negLinearSVC,'-go',label='LinearSVC')
    lin2, = ax[1,1].plot(np.arange(1,30),np.ones(29) * false_neg_tLinearSVC,'-gx',label='LinearSVC train')
    sgd1, = ax[1,1].plot(false_n,'-bo',label='SGDClassifier')
    sgd2, = ax[1,1].plot(false_n_t,'-bx',label='SGDClassifier train')
    ax[1, 1].set_title('False Negatives')
    
    svc1, = ax[1,0].plot(np.arange(1,30),np.ones(29) * false_pos_SVC,'-ro',label='SVC')
    svc2, =ax[1,0].plot(np.arange(1,30),np.ones(29) * false_pos_t_SVC,'-rx',label='SVC train')
    lin1, = ax[1,0].plot(np.arange(1,30),np.ones(29) * false_posLinearSVC,'-go',label='LinearSVC')
    lin2, = ax[1,0].plot(np.arange(1,30),np.ones(29) * false_pos_tLinearSVC,'-gx',label='LinearSVC train')
    sgd1, = ax[1,0].plot(false_p,'-bo',label='SGDClassifier')
    sgd2, = ax[1,0].plot(false_p_t,'-bx',label='SGDClassifier train')
    ax[1, 0].set_title('False Positives')
    
    
    fig.legend((svc1,svc2,lin1,lin2,sgd1,sgd2),('SVC','SVC_t','Lin','Lin_t','SGD','SGD_t'), loc='upper right', numpoints = 1 , prop={'size':5})
    fig.suptitle('%s'%(class_))
    plt.savefig('/home/tstahl/classifiers/%s_classifier_hybrid.png'%(class_))
    plt.clf()
    
    # find best classifier:
    correct = np.array(true_n_t) + np.array(true_p_t)
    maximum = np.argmax(correct)
    print 'SGD: ', np.array(true_n) + np.array(true_p), maximum
    print 'SVMs: ', true_posLinearSVC + true_negLinearSVC, true_pos_SVC + true_neg_SVC
    
    # classifier picked always predicts zeros (i think it's SVC)
    clf = clfs[maximum]
#    if correct[maximum] >= true_pos_tLinearSVC + true_neg_tLinearSVC  and correct[maximum] >= true_pos_t_SVC + true_neg_t_SVC:
#        clf = clfs[maximum]
#    else:
#        if true_pos_tLinearSVC + true_neg_tLinearSVC > true_pos_t_SVC + true_neg_t_SVC:
#            clf = clf2
#        else:
#            clf = clf1
    for epochs in np.arange(1,5):
        for levels_num in np.arange(1,6):
            global prune_tree_levels
            prune_tree_levels = levels_num
            if normalize:
                if os.path.isfile('/var/node436/local/tstahl/Models/'+class_+c+'%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_scaler.pickle'%(learning_rate0, alphas[0],alphas[1],learn_intersections,subsampling,squared_hinge_loss,prune_fully_covered,reg,mous,prune_tree_levels)):
                    with open('/var/node436/local/tstahl/Models/'+class_+c+'%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_scaler.pickle'%(learning_rate0, alphas[0],alphas[1],learn_intersections,subsampling,squared_hinge_loss,prune_fully_covered,reg,mous,prune_tree_levels), 'r') as handle:
                        scaler = pickle.load(handle) 
                        print 'scaler loaded'
                else:
                    print 'normalizing scaler'
                    scaler = MinMaxScaler()
                    for minibatch in range(0,len(train_imgs)):
                        sum_x,n_samples,sum_sq_x,scaler = minibatch_(None,None,scaler,[], [],[],[],[],[],[],[],test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,None,None,'mean_variance')
                        
            for alpha1 in all_alphas:
                for reg in regs:
                    for learning_rate0 in learning_rates:
                        learning_rate = learning_rate0
                        alphas = [1-alpha1,alpha1,reg]
                        if less_features:
                            w = 0.01 * np.random.rand(features_used)
                        else:
                            w = 0.01 * np.random.rand(4096)                    
                        loss_train = []
                        loss_test = []
                        full_image_test = []
                        full_image_train = []
                        learning_rate_again = []
                        t = 0
                        start = time.time()    
                        
                        mse_train_ = []
                        mse_test_ = []
                        
                        mse_mxlvl_train = []
                        mse_mxlvl_test = []
                        
                        mse_fllycover_train = []
                        mse_fllycover_test = []
                        
                        clte = []
                        cltr = []
                        
                        lmte = []
                        lmtr = []
                        if os.path.isfile('/var/node436/local/tstahl/Models/'+class_+c+'%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_hybrid.pickle'%(learning_rate0, alphas[0],alphas[1],learn_intersections,subsampling,squared_hinge_loss,prune_fully_covered,reg,mous,prune_tree_levels,epochs)):
                            with open('/var/node436/local/tstahl/Models/'+class_+c+'%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_hybrid.pickle'%(learning_rate0, alphas[0],alphas[1],learn_intersections,subsampling,squared_hinge_loss,prune_fully_covered,reg,mous,prune_tree_levels,epochs), 'r') as handle:
                                w = pickle.load(handle)
                                print 'model exists, loaded'
                                new = False
                        else:
                            new = True
                            print 'training model'
                            loss__train = []
                            full_image__train = []
                            for epoch in range(epochs):
                                print epoch, learning_rate, alphas
                                #shuffle images, not boxes!
                                random.shuffle(shuffled)
                                for it,minibatch in enumerate(shuffled):
                                    w,le,nr_nodes = minibatch_(functions,None,scaler,w, [],[],[],[],[],alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance,'train')
                                    t += nr_nodes
                                    print it,', # boxes: ', t, ' used: ', nr_nodes
                                learning_rate = learning_rate0 * (1+learning_rate0*gamma*t)**-1
                                # check for convergence
                                loss__train = []
                                print 'compute loss'
                                for minibatch in range(0,len(train_imgs)):
                                    loss__train = minibatch_(functions,clf,scaler,w, loss__train,[],[],[],full_image__train,alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance,'loss_train')
                                # check for convergence
                                new_loss = np.array(loss__train).sum()
                                plot_loss.append(new_loss)
                                if new_loss < converged:
                                    if converged - new_loss < -1:
                                        break
                                    else:
                                        converged = new_loss
                                else:
                                    converged = new_loss
                        learning_rate_again.append(learning_rate)
                        level_mse_train = []
                        level_mse_test = []
                        full_cover_level_score_train = []
                        full_cover_level_score_test = []
                        full_cover_level_score_train_nn = []
                        full_cover_level_score_test_nn = []
                        max_level_score_train = []
                        max_level_score_test = []
                        max_level_score_train_nn = []
                        max_level_score_test_nn = []
                        covered_levels_train = []
                        covered_levels_test = []
                        loss__train = []
                        loss__test = []
                        mse__train = []
                        mse__test = []
                        full_image__train = []
                        full_image__test= []
                        median_test = []
                        median_test_nn = []
                        min_level_test = []
                        min_level_test_nn = []
                        just_level = {}
                        just_level_nn = {}
                        plt.figure()
                        plt.plot(plot_loss, '-r')
                        plt.savefig('/home/tstahl/%s_%s_loss_prop_inter.png'%(class_, prune_tree_levels))
                        plt.clf()
        
                        for minibatch in range(0,len(test_imgs)):
                            loss__test,full_image__test = minibatch_(functions,clf,scaler,w, loss__test,[],[],[],full_image__test,alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance,'loss_test')
                            cpls,trew = minibatch_(functions, clf,scaler,w, [],[],[],[],[],alphas,learning_rate0,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance, 'levels_test')
                            max_level_score_test.append(max(cpls))                        
                            full_cover_level_score_test.append(np.array(cpls).sum()/len(cpls))
                            median_test.append(np.median(cpls))
                            min_level_test.append(min(cpls))
                            for i,cpl in enumerate(cpls):
                                if i in just_level:
                                    just_level[i].append(cpl)
                                else: 
                                    just_level[i] = [cpl]
                            if len(cpls) < prune_tree_levels:
                                for lev in range(len(cpls),prune_tree_levels):
                                    if lev in just_level:
                                        just_level[lev].append(cpls[-1])
                                    else:
                                        just_level[lev] = [cpls[-1]]
                            if trew[0] > 0:                 
                                max_level_score_test_nn.append(max(cpls))
                                full_cover_level_score_test_nn.append(np.array(cpls).sum()/len(cpls))
                                median_test_nn.append(np.median(cpls))
                                min_level_test_nn.append(min(cpls))
                                for i,cpl in enumerate(cpls):
                                    if i in just_level_nn:
                                        just_level_nn[i].append(cpl)
                                    else: 
                                        just_level_nn[i] = [cpl]
                                if len(cpls) < prune_tree_levels:
                                    for lev in range(len(cpls),prune_tree_levels):
                                        if lev in just_level_nn:
                                            just_level_nn[lev].append(cpls[-1])
                                        else:
                                            just_level_nn[lev] = [cpls[-1]]
                            level_mse_test = np.hstack((level_mse_test,((np.array(cpls)-np.array(trew))**2)))
                            covered_levels_test = np.hstack((covered_levels_test,((np.array(cpls)-np.array(trew[0:len(cpls)]))**2)))
                        
        
                        # save avg loss for plotting
        #                        loss_train.append(sum(loss__train)/len(loss__train))
                        loss_test.append(sum(loss__test)/len(loss__test))
        #                        full_image_train.append(((np.array([z[0] for z in full_image__train]) - np.array( [z_[1] for z_ in full_image__train]))**2).sum() / len(full_image__train))
                        clte.append((np.array(covered_levels_test).sum()/len(covered_levels_test)))
        #                        cltr.append((np.array(covered_levels_train).sum()/len(covered_levels_train)))
                        lmte.append((np.array(level_mse_test).sum()/len(level_mse_test)))
        #                        lmtr.append((np.array(level_mse_train).sum()/len(level_mse_train)))
                        
        #                        mean_preds_train = np.array( [z_[1] for z_ in full_image__train]).sum() / len(full_image__train)
        #                        mean_obs_train = np.array( [z_[0] for z_ in full_image__train]).sum() / len(full_image__train)
                        mean_preds_test = np.array( [z_[1] for z_ in full_image__test]).sum() / len(full_image__test)
                        mean_obs_test = np.array( [z_[0] for z_ in full_image__test]).sum() / len(full_image__test)
                        
        #                        mean_preds_train_mx_lvl = np.array(max_level_score_train).sum()/len(max_level_score_train)
        #                        mean_preds_train_cov_lvl = np.array(full_cover_level_score_train).sum() / len(full_cover_level_score_train)
                        mean_preds_test_mx_lvl = np.array(max_level_score_test).sum() / len(max_level_score_test)
                        mean_preds_test_cov_lvl = np.array(full_cover_level_score_test).sum() / len(full_cover_level_score_test)
                        mean_preds_test_min_lvl = np.array(min_level_test).sum() / len(min_level_test)
                        mean_preds_test_median = np.array(median_test).sum() / len(median_test)
                        mean_just_level = {}
                        for i,lvl in enumerate(just_level):
                            mean_just_level[i] = np.array(just_level[i]).sum() / len(just_level[i])
                        
                        
        #                        nmse_train_ = (((np.array([z[0] for z in full_image__train]) -np.array( [z_[1] for z_ in full_image__train]))**2).sum() / len(full_image__train) / (mean_preds_train * mean_obs_train))
        #                        nmse_train_mx_lvl = (((np.array([z[0] for z in full_image__train]) -np.array( max_level_score_train))**2).sum() / len(full_image__train) / (mean_preds_train_mx_lvl * mean_obs_train))
        #                        nmse_train_cov_lvl = (((np.array([z[0] for z in full_image__train]) -np.array( full_cover_level_score_train))**2).sum() / len(full_image__train) / (mean_preds_train_cov_lvl * mean_obs_train))
                        nmse_test_ = (((np.array([z[0] for z in full_image__test]) - np.array([z_[1] for z_ in full_image__test]))**2).sum() / len(full_image__test) / (mean_preds_test * mean_obs_test))
                        nmse_test_mx_lvl = (((np.array([z[0] for z in full_image__test]) - np.array(max_level_score_test))**2).sum() / len(full_image__test) / (mean_preds_test_mx_lvl * mean_obs_test))
                        nmse_test_cov_lvl = (((np.array([z[0] for z in full_image__test]) - np.array(full_cover_level_score_test))**2).sum() / len(full_image__test) / (mean_preds_test_cov_lvl * mean_obs_test))
                        nmse_preds_test_min_lvl = ((np.array([z[0] for z in full_image__test]) - np.array(min_level_test))**2).sum() / len(min_level_test) / (mean_obs_test * mean_preds_test_min_lvl)
                        nmse_preds_test_median = ((np.array([z[0] for z in full_image__test]) - np.array(median_test))**2).sum() / len(median_test) / (mean_obs_test * mean_preds_test_median)
                        nmse_just_level = {}
                        for i,lvl in enumerate(just_level):
                            nmse_just_level[i] = ((np.array([z[0] for z in full_image__test]) - np.array(just_level[i]))**2).sum() / len(just_level[i]) / (mean_obs_test * mean_just_level[i])
                        
        #                        mse_train_ = (((np.array([z[0] for z in full_image__train]) -np.array( [z_[1] for z_ in full_image__train]))**2).sum() / len(full_image__train))
                        mse_test_ = (((np.array([z[0] for z in full_image__test]) - np.array([z_[1] for z_ in full_image__test]))**2).sum() / len(full_image__test))
        #                        print nmse_train_, nmse_test_
                        #raw_input()
        #                        mse_mxlvl_train = ((np.array([z[0] for z in full_image__train]) -np.array(max_level_score_train ))**2).sum() / len(full_image__train)
                        mse_mxlvl_test = ((np.array([z[0] for z in full_image__test]) - np.array(max_level_score_test))**2).sum() / len(full_image__test)
        #                        mse_fllycover_train = ((np.array([z[0] for z in full_image__train]) -np.array(full_cover_level_score_train ))**2).sum() / len(full_image__train)
                        mse_fllycover_test = ((np.array([z[0] for z in full_image__test]) - np.array(full_cover_level_score_test))**2).sum() / len(full_image__test)
                        
                        mse_preds_test_min_lvl = ((np.array([z[0] for z in full_image__test]) - np.array(min_level_test))**2).sum() / len(min_level_test)
                        mse_preds_test_median = ((np.array([z[0] for z in full_image__test]) - np.array(median_test))**2).sum() / len(median_test)
                        mse_just_level = {}
                        for i,lvl in enumerate(just_level):
                            mse_just_level[i] = ((np.array([z[0] for z in full_image__test]) - np.array(just_level[i]))**2).sum() / len(just_level[i])
        #                        mse_train_nn = (((np.array([z[0] for z in full_image__train if z[0]>0]) -np.array( [z_[1] for z_ in full_image__train if z_[0]>0]))**2).sum() / len([z for z in full_image__train if z[0]>0]))
                        mse_test_nn = (((np.array([z[0] for z in full_image__test if z[0]>0]) - np.array([z_[1] for z_ in full_image__test if z_[0]>0]))**2).sum() / len([z for z in full_image__test if z[0]>0]))
        #                        mse_mxlvl_train_nn = ((np.array([z[0] for z in full_image__train if z[0]>0]) -np.array(max_level_score_train_nn ))**2).sum() / len(max_level_score_train_nn)
                        mse_mxlvl_test_nn = ((np.array([z[0] for z in full_image__test if z[0]>0]) - np.array(max_level_score_test_nn))**2).sum() / len(max_level_score_test_nn)
        #                        mse_fllycover_train_nn = ((np.array([z[0] for z in full_image__train if z[0]>0]) -np.array(full_cover_level_score_train_nn ))**2).sum() / len(full_cover_level_score_train_nn)
                        mse_fllycover_test_nn = ((np.array([z[0] for z in full_image__test if z[0]>0]) - np.array(full_cover_level_score_test_nn))**2).sum() / len(full_cover_level_score_test_nn)                
                        
                        mse_preds_test_min_lvl_nn = ((np.array([z[0] for z in full_image__test if z[0]>0]) - np.array(min_level_test_nn))**2).sum() / len(min_level_test_nn)
                        mse_preds_test_median_nn = ((np.array([z[0] for z in full_image__test if z[0]>0]) - np.array(median_test_nn))**2).sum() / len(median_test_nn)
                        mse_just_level_nn = {}
                        for i,lvl in enumerate(just_level_nn):
                            mse_just_level_nn[i] = ((np.array([z[0] for z in full_image__test if z[0]>0]) - np.array(just_level_nn[i]))**2).sum() / len(just_level_nn[i])
                        
        #                        mean_preds_train_nn = np.array( [z_[1] for z_ in full_image__train if z_[0]>0]).sum() / len([z for z in full_image__train if z[0]>0])
        #                        print len([z_[1] for z_ in full_image__train if z_[0]>0]), len([z for z in full_image__train if z[0]>0])
        #                        mean_obs_train_nn = np.array( [z_[0] for z_ in full_image__train if z_[0]>0]).sum() / len([z for z in full_image__train if z[0]>0])
                        mean_preds_test_nn = np.array( [z_[1] for z_ in full_image__test if z_[0]>0]).sum() / len([z for z in full_image__test if z[0]>0])
                        mean_obs_test_nn = np.array( [z_[0] for z_ in full_image__test if z_[0]>0]).sum() / len([z for z in full_image__test if z[0]>0])
                        
        #                        mean_preds_train_mx_lvl_nn = np.array(max_level_score_train_nn).sum()/len(max_level_score_train_nn)
        #                        mean_preds_train_cov_lvl_nn = np.array(full_cover_level_score_train_nn).sum() / len(full_cover_level_score_train_nn)
                        mean_preds_test_mx_lvl_nn = np.array(max_level_score_test_nn).sum() / len(max_level_score_test_nn)
                        mean_preds_test_cov_lvl_nn = np.array(full_cover_level_score_test_nn).sum() / len(full_cover_level_score_test_nn)
                        mean_preds_test_min_lvl_nn = np.array(min_level_test_nn).sum() / len(min_level_test_nn)
                        mean_preds_test_median_nn = np.array(median_test_nn).sum() / len(median_test_nn)
                        mean_just_level_nn = {}
                        for i,lvl in enumerate(just_level_nn):
                            mean_just_level_nn[i] = np.array(just_level_nn[i]).sum() / len(just_level_nn[i])
                        
        #                        nmse_train_nn = mse_train_nn / (mean_preds_train_nn * mean_obs_train_nn)
        #                        nmse_train_mx_lvl_nn = mse_mxlvl_train_nn / (mean_preds_train_mx_lvl_nn * mean_obs_train_nn)
        #                        nmse_train_cov_lvl_nn = mse_fllycover_train_nn / (mean_preds_train_cov_lvl_nn * mean_obs_train_nn)
                        nmse_test_nn = mse_test_nn / (mean_preds_test_nn * mean_obs_test_nn)
                        nmse_test_mx_lvl_nn = mse_mxlvl_test_nn / (mean_preds_test_mx_lvl_nn * mean_obs_test_nn)
                        nmse_test_cov_lvl_nn = mse_fllycover_test_nn / (mean_preds_test_cov_lvl_nn * mean_obs_test_nn)
                        nmse_preds_test_min_lvl_nn = mse_preds_test_min_lvl_nn / len(min_level_test_nn) / (mean_obs_test_nn * mean_preds_test_min_lvl_nn)
                        nmse_preds_test_median_nn = mse_preds_test_median_nn / len(median_test_nn) / (mean_obs_test_nn * mean_preds_test_median_nn)
                        nmse_just_level_nn = {}
                        for i,lvl in enumerate(just_level_nn):
                            nmse_just_level_nn[i] = mse_just_level_nn[i] / len(just_level_nn[i]) / (mean_obs_test_nn * mean_just_level_nn[i])
                
        
                        text = ''
                        for i in range(prune_tree_levels):
                            text += 'Lvl %s:%s(%s)[%s(%s)]\n' %(i,round(mse_just_level[i],2),round(nmse_just_level[i],2),round(mse_just_level_nn[i],2),round(nmse_just_level_nn[i],2))
                        plt.xlabel('ground truth')
                        plt.ylabel('predictions')
        #                    plt.plot([z[0] for z in full_image__train], [z_[1] for z_ in full_image__train],'rx', label='full image train')
                        plt.plot([z[0] for z in full_image__test], [z_[1] for z_ in full_image__test],'ro',label='full image test')
        #                    plt.plot([z[0] for z in full_image__train], max_level_score_train,'gx', label='max level train')
                        plt.plot([z[0] for z in full_image__test], max_level_score_test,'go',label='max level test')
                        plt.plot(range(28),range(28),'k-')
        #                    plt.plot([z[0] for z in full_image__train], full_cover_level_score_train,'yx', label='fully cover level train')
                        plt.plot([z[0] for z in full_image__test], full_cover_level_score_test,'yo',label='fully cover level test')
                        plt.xticks(range(0,31,5))
                        leg = plt.legend( loc='upper left', numpoints = 1 , prop={'size':8})
                        leg.get_frame().set_alpha(0.5)
                        plt.text(25,15,'MSE test:\n' + str(round(mse_test_,2))+'('+ str(round(nmse_test_,2)) +')' +'['+str(round(mse_test_nn,2)) +'('+str(round(nmse_test_nn,2))+')'+']'+'\nMSE minlvl test:\n' + str(round(mse_preds_test_min_lvl,2)) +'('+str(round(nmse_preds_test_min_lvl,2))+')' +'['+str(round(mse_preds_test_min_lvl_nn,2)) +'('+str(round(nmse_preds_test_min_lvl_nn,2))+')'+']'+ '\nMSE mxlvl test:\n' + str(round(mse_mxlvl_test,2))+'('+str(round(nmse_test_mx_lvl,2))+')'+'['+str(round(mse_mxlvl_test_nn,2)) +'('+str(round(nmse_test_mx_lvl_nn,2))+')'+']'+'\nMSE median test:\n' + str(round(mse_preds_test_median,2)) +'('+str(round(nmse_preds_test_median,2))+')'+'['+str(round(mse_preds_test_median_nn,2)) +'('+str(round(nmse_preds_test_median_nn,2))+')'+']'+ '\nMSE fllycover test:\n' + str(round(mse_fllycover_test,2))+'('+str(round(nmse_test_cov_lvl,2))+')'+'['+str(round(mse_fllycover_test_nn,2)) +'('+str(round(nmse_test_cov_lvl_nn,2))+')'+']\n'+text, verticalalignment='bottom',
                                 horizontalalignment='left',
                                 fontsize=8,
                                 bbox={'facecolor':'white', 'alpha':0.6, 'pad':10})
                        #plt.title('Predictions full image:%s,%s,%s,%s\nmse tree levels train: %s, mse tree levels test: %s\nmse covered levels train:%s,mse covered levels test:%s '%(class_,learning_rate0,alphas[0], alphas[1],lmtr[-1],lmte[-1],cltr[-1],clte[-1]),fontsize=8)
                        plt.savefig('/home/tstahl/full_image_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_clipped_more_hybrid.png'%(class_,learning_rate0, alphas[0],alphas[1],learn_intersections,subsampling,squared_hinge_loss,prune_fully_covered,reg,mous,prune_tree_levels,jans_idea,epochs))
                        print "model learned with learning rate: %s, alphas: %s, subsampling: %s, learn_intersections: %s, squared hinge loss 2nd constraint: %s "%(learning_rate0,alphas,subsampling,learn_intersections, squared_hinge_loss)
                        print new                    
                        if new:                    
                            with open('/var/node436/local/tstahl/Models/'+class_+c+'%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_hybrid.pickle'%(learning_rate0, alphas[0],alphas[1],learn_intersections,subsampling,squared_hinge_loss,prune_fully_covered,reg,mous,prune_tree_levels,epochs), 'wb') as handle:
                                pickle.dump(w, handle)
                            with open('/var/node436/local/tstahl/Models/'+class_+c+'%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_scaler.pickle'%(learning_rate0, alphas[0],alphas[1],learn_intersections,subsampling,squared_hinge_loss,prune_fully_covered,reg,mous,prune_tree_levels), 'wb') as handle:
                                pickle.dump(scaler, handle)
                        with open('/var/node436/local/tstahl/Ims/'+class_+c+'_%s_%s_mean_hybrid.pickle'%(prune_tree_levels,epochs), 'wb') as handle:
                                pickle.dump(mean_just_level, handle)
                        with open('/var/node436/local/tstahl/Ims/'+class_+c+'_%s_%s_mean_nn_hybrid.pickle'%(prune_tree_levels,epochs), 'wb') as handle:
                                pickle.dump(mean_just_level_nn, handle)
                        with open('/var/node436/local/tstahl/Ims/'+class_+c+'_%s_%s_mse_hybrid.pickle'%(prune_tree_levels,epochs), 'wb') as handle:
                                pickle.dump(mse_just_level, handle)
                        with open('/var/node436/local/tstahl/Ims/'+class_+c+'_%s_%s_mse_nn_hybrid.pickle'%(prune_tree_levels,epochs), 'wb') as handle:
                                pickle.dump(mse_just_level_nn, handle)
                        plt.clf()


def bool_rect_intersect(A, B):
    return not (B[0]>A[2] or B[2]<A[0] or B[3]<A[1] or B[1]>A[3]), 1/((A[2]- A[0] + 1)*(A[3]-A[1] + 1))
        #return !(r2.left > r1.right || r2.right < r1.left || r2.top > r1.bottom ||r2.bottom < r1.top);
    
if __name__ == "__main__":
#    cProfile.run('main()')
    main()
