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
from load import get_seperation, get_data,get_image_numbers,get_class_data, get_traineval_seperation, get_data_from_img_nr
import matplotlib.image as mpimg
from utils import create_tree, find_children, sort_boxes, surface_area, extract_coords, get_set_intersection
from ml import tree_level_regression, tree_level_loss, count_per_level, sums_of_all_cliques, constrained_regression, learn_root, loss
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


def minibatch_(functions, clf,scaler,w, loss__,mse,hinge1,hinge2,full_image,img_nr,alphas,learning_rate,subsamples, mode):
    X_p, y_p, inv = get_data_from_img_nr(class_,img_nr, subsamples)
    if X_p != []:
        boxes = []
        ground_truth = inv[0][2]
        img_nr = inv[0][0]
        print img_nr
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
        levels_tmp = {k:v for k,v in levels.iteritems() if k<prune_tree_levels}
        levels_gone = {k:v for k,v in levels.iteritems() if k>=prune_tree_levels}
        levels = levels_tmp
        #prune tree as well, for patches training
        for trash_level in levels_gone.values():
            G.remove_nodes_from(trash_level)
        
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
        if mode == 'training':
            if alphas[0] == 0: #if we don't learn the proposals, we learn just the levels: better, because every level has same importance and faster
                print 'training levels', img_nr
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
                print 'training patches', img_nr
                print predecs
                nodes = list(G.nodes())
                for node in nodes:
                    print node
                    if node == 0:
                        w = learn_root(w,norm_x[0],pruned_y[0],learning_rate,alphas)
                    else:
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
                        #w, function = tree_level_regression(class_,function,levels,level,features,coords,scaler,w,norm_x,pruned_y,node,predecs,children,pruned_boxes,learning_rate,alphas,img_nr)
                        w, function = constrained_regression(class_,function,features,coords,scaler,w,norm_x,pruned_y,node,predecs,children,pruned_boxes,learning_rate,alphas,img_nr,squared_hinge_loss)
                        #TODO: train regressor/classifier that predicts/chooses level. Features: level, number of proposals, number of intersections, avg size of proposal, predictions(for regressor), etc.
                        if level not in functions[img_nr]:
                            functions[img_nr][level] = function
                return w, len(pruned_y), len(G.nodes())
        elif mode == 'scikit_train':
            clf.partial_fit(norm_x,pruned_y)
            return clf
        elif mode == 'loss_train':
            if alphas[0] == 0: #levels
                loss__.append(tree_level_loss(class_,features,coords,scaler, w, data, predecs, children,img_nr,-1,functions))
                return loss__
            else:
                loss__.append(loss(class_,squared_hinge_loss,features,coords,scaler,w, data, predecs, children,img_nr, -1))
        elif mode == 'loss_test' or mode == 'loss_eval':
            print mode, loss__
            if alphas[0] == 0: #levels
                loss__.append(tree_level_loss(class_,features,coords,scaler, w, data, predecs, children,img_nr,-1,functions))
                cpl = max(0, np.dot(w,np.array(norm_x[0]).T))
                full_image.append([pruned_y[0],cpl])
                return loss__,full_image
            else:
                loss__.append(loss(class_,squared_hinge_loss,features,coords,scaler,w, data, predecs, children,img_nr, -1))
                cpl = max(0, np.dot(w,np.array(norm_x[0]).T))
                full_image.append([pruned_y[0],cpl])
                return loss__,full_image
        elif mode == 'loss_scikit_test' or mode == 'loss_scikit_train':
            loss__.append(((clf.predict(norm_x) - pruned_y)**2).sum())
            return loss__ 
        elif mode == 'levels_train' or mode == 'levels_test':
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
    train_imgs = train_imgs[0:100]
    test_imgs = test_imgs[0:100]
    training_imgs, evaluation_imgs = get_traineval_seperation(train_imgs)
    # learn
#    if os.path.isfile('/var/node436/local/tstahl/Models/'+class_+c+'normalized_constrained.pickle'):
#        with open('/var/node436/local/tstahl/Models/'+class_+c+'normalized_constrained.pickle', 'rb') as handle:
#            w = pickle.load(handle)
#    else:
    evaluation_imgs = evaluation_imgs[0:100]
    print training_imgs, evaluation_imgs, test_imgs
    img_train = training_imgs
    img_eval = evaluation_imgs
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
    shuffled = training_imgs

     
    #normalize
    learning_rate0 = learning_rates[0]
    alpha1 =         all_alphas[0]
    reg = regs[0]
    alphas_levels = [1-alpha1,alpha1,reg]
    alphas_patches = [alpha1,reg,alpha1,alpha1]
    alphas_just_patches = [1, reg, 0, 0]
    alphas_just_parent = [1, reg, 1, 0]
    alphas_just_level = [1, reg, 0, 1]
    X_ = []
    y_ = []
    X_test = []
    y_test = []
    w_all = {}
    for levels_num in np.arange(2,4):
        learning_rate = learning_rate0
        for epochs in np.arange(1,4):
            global prune_tree_levels
            prune_tree_levels = levels_num
            if normalize:
                if os.path.isfile('/var/node436/local/tstahl/Models/'+class_+c+'%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_scaler.pickle'%(learning_rate0, alphas_levels[0],alphas_levels[1],learn_intersections,subsampling,squared_hinge_loss,prune_fully_covered,reg,mous,prune_tree_levels)):
                    with open('/var/node436/local/tstahl/Models/'+class_+c+'%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_scaler.pickle'%(learning_rate0, alphas_levels[0],alphas_levels[1],learn_intersections,subsampling,squared_hinge_loss,prune_fully_covered,reg,mous,prune_tree_levels), 'r') as handle:
                        scaler = pickle.load(handle) 
                        print 'scaler loaded'
                else:
                    print 'normalizing scaler'
                    scaler = MinMaxScaler()
                    for img_nr in img_train:
                        sum_x,n_samples,sum_sq_x,scaler = minibatch_(None,None,scaler,[], [],[],[],[],[],img_nr,[],[],subsamples,'mean_variance')
            alphas = [1-alpha1,alpha1,reg]
            
            if epochs == 1:
                # initialize or reset w , plot_losses
                w_all[levels_num] = []
                if less_features:
                    w_levels = 0.01 * np.random.rand(features_used)
                    w_patches = 0.01 * np.random.rand(features_used)
                else:
                    w_levels = 0.01 * np.random.rand(4096)     
                    w_patches = 0.01 * np.random.rand(4096)     
                    w_just_patches = 0.01 * np.random.rand(4096)
                    w_just_parent = 0.01 * np.random.rand(4096)
                    w_just_level = 0.01 * np.random.rand(4096)
                plot_training_loss_levels = []
                plot_evaluation_loss_levels = []
                plot_training_loss_patches = []
                plot_evaluation_loss_patches = []
            loss_train= []
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
            new = True
            print 'training model'
            loss_train_levels = []
            loss_eval_levels = []
            loss_train_patches = []
            loss_eval_patches = []
            full_image__train = []
            print epochs, learning_rate, alphas
            #shuffle images, not boxes!
            random.shuffle(shuffled)
            for img_nr in shuffled:
                print img_nr
                w_levels,le,nr_nodes = minibatch_(functions,None,scaler,w_levels, [],[],[],[],[],img_nr,alphas_levels,learning_rate,subsamples,'training')
                w_patches,le,nr_nodes = minibatch_(functions,None,scaler,w_patches, [],[],[],[],[],img_nr,alphas_patches,learning_rate,subsamples,'training')
                w_just_patches,_,_ = minibatch_(functions,None,scaler,w_just_patches, [],[],[],[],[],img_nr,alphas_just_patches,learning_rate,subsamples,'training')
                w_just_parent,_,_ = minibatch_(functions,None,scaler,w_just_parent, [],[],[],[],[],img_nr,alphas_just_parent,learning_rate,subsamples,'training')
                w_just_level,_,_ = minibatch_(functions,None,scaler,w_just_level, [],[],[],[],[],img_nr,alphas_just_level,learning_rate,subsamples,'training')
                t += nr_nodes
                print ', # boxes: ', t, ' used: ', nr_nodes
            learning_rate = learning_rate0 * (1+learning_rate0*gamma*t)**-1
            
            # check for convergence
            loss_train_levels = []
            print 'compute loss'
            for img_nr in training_imgs:
                loss_train_levels = minibatch_(functions,[],scaler,w_levels, loss_train_levels,[],[],[],full_image__train,img_nr,alphas_levels,learning_rate,subsamples,'loss_training')
                loss_train_patches = minibatch_(functions,[],scaler,w_patches, loss_train_patches,[],[],[],full_image__train,img_nr,alphas_patches,learning_rate,subsamples,'loss_training')
            for img_nr in evaluation_imgs:
                loss_eval_levels,_ = minibatch_(functions,[],scaler,w_levels, loss_eval_levels,[],[],[],full_image__train,img_nr,alphas_levels,learning_rate,subsamples,'loss_eval')
                #print loss_eval_patches
                loss_eval_patches,_ = minibatch_(functions,[],scaler,w_levels, loss_eval_patches,[],[],[],full_image__train,img_nr,alphas_patches,learning_rate,subsamples,'loss_eval')
            

            plot_training_loss_levels.append(np.array(loss_train_levels).sum())
            plot_evaluation_loss_levels.append(np.array(loss_eval_levels).sum())
            plot_training_loss_patches.append(np.array(loss_train_patches).sum())
            plot_evaluation_loss_patches.append(np.array(loss_eval_patches).sum())
                        
                            
        # Training done
        w_all[levels_num].append(w_levels)
        #last epoch, save training, validation loss etc.
        plt.figure()
        plt.plot(plot_training_loss_levels, '-.r',label="Levels train")
        plt.plot(plot_evaluation_loss_levels, '-r',label="Levels eval")
        plt.plot(plot_training_loss_patches, '-.b',label="Patches train")
        plt.plot(plot_evaluation_loss_patches, '-b',label="Patches eval")
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.savefig('/home/tstahl/%s_%s_loss_prop_inter.png'%(class_, prune_tree_levels))
        plt.legend()
        plt.clf()
                        
        
        for name, w, alpha in zip(['levels','patches_all','just_patches','just_parent','just_level'],[w_levels, w_patches, w_just_patches, w_just_parent, w_just_level],[alphas_levels,alphas_patches, alphas_just_patches, alphas_just_parent, alphas_just_level]):
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
    
            for img_nr in test_imgs:
                loss__test,full_image__test = minibatch_(functions,None,scaler,w, loss__test,[],[],[],full_image__test,img_nr,alphas,learning_rate,subsamples,'loss_test')
                cpls,trew = minibatch_(functions, [],scaler,w, [],[],[],[],[],img_nr,alphas,learning_rate0,subsamples, 'levels_test')
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
            
            print name
            print 'MSE test:\n' + str(round(mse_test_,2))+'('+ str(round(nmse_test_,2)) +')' +'['+str(round(mse_test_nn,2)) +'('+str(round(nmse_test_nn,2))+')'+']'+'\nMSE minlvl test:\n' + str(round(mse_preds_test_min_lvl,2)) +'('+str(round(nmse_preds_test_min_lvl,2))+')' +'['+str(round(mse_preds_test_min_lvl_nn,2)) +'('+str(round(nmse_preds_test_min_lvl_nn,2))+')'+']'+ '\nMSE mxlvl test:\n' + str(round(mse_mxlvl_test,2))+'('+str(round(nmse_test_mx_lvl,2))+')'+'['+str(round(mse_mxlvl_test_nn,2)) +'('+str(round(nmse_test_mx_lvl_nn,2))+')'+']'+'\nMSE median test:\n' + str(round(mse_preds_test_median,2)) +'('+str(round(nmse_preds_test_median,2))+')'+'['+str(round(mse_preds_test_median_nn,2)) +'('+str(round(nmse_preds_test_median_nn,2))+')'+']'+ '\nMSE fllycover test:\n' + str(round(mse_fllycover_test,2))+'('+str(round(nmse_test_cov_lvl,2))+')'+'['+str(round(mse_fllycover_test_nn,2)) +'('+str(round(nmse_test_cov_lvl_nn,2))+')'+']\n'+text
            text_file = open("/home/tstahl/%s/%s_%s.txt"%(class_,name, levels_num), "w")
            text_file.write('MSE test:\n' + str(round(mse_test_,2))+'('+ str(round(nmse_test_,2)) +')' +'['+str(round(mse_test_nn,2)) +'('+str(round(nmse_test_nn,2))+')'+']'+'\nMSE minlvl test:\n' + str(round(mse_preds_test_min_lvl,2)) +'('+str(round(nmse_preds_test_min_lvl,2))+')' +'['+str(round(mse_preds_test_min_lvl_nn,2)) +'('+str(round(nmse_preds_test_min_lvl_nn,2))+')'+']'+ '\nMSE mxlvl test:\n' + str(round(mse_mxlvl_test,2))+'('+str(round(nmse_test_mx_lvl,2))+')'+'['+str(round(mse_mxlvl_test_nn,2)) +'('+str(round(nmse_test_mx_lvl_nn,2))+')'+']'+'\nMSE median test:\n' + str(round(mse_preds_test_median,2)) +'('+str(round(nmse_preds_test_median,2))+')'+'['+str(round(mse_preds_test_median_nn,2)) +'('+str(round(nmse_preds_test_median_nn,2))+')'+']'+ '\nMSE fllycover test:\n' + str(round(mse_fllycover_test,2))+'('+str(round(nmse_test_cov_lvl,2))+')'+'['+str(round(mse_fllycover_test_nn,2)) +'('+str(round(nmse_test_cov_lvl_nn,2))+')'+']\n'+text)
            text_file.close()

def bool_rect_intersect(A, B):
    return not (B[0]>A[2] or B[2]<A[0] or B[3]<A[1] or B[1]>A[3]), 1/((A[2]- A[0] + 1)*(A[3]-A[1] + 1))
        #return !(r2.left > r1.right || r2.right < r1.left || r2.top > r1.bottom ||r2.bottom < r1.top);
    
if __name__ == "__main__":
#    cProfile.run('main()')
    main()
