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
from train_per_level import train_per_level
import matplotlib.colors as colors
import matplotlib.image as mpimg
import time
import matplotlib.cm as cmx
from scipy import optimize
import time
import cProfile
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from matplotlib.patches import Rectangle

class_ = 'sheep'
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
squared_hinge_loss = True



class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
### this traines the model; uses an extra feature - window size

def find_children(sucs, parent):
    if parent < len(sucs):
        if sucs.keys()[parent] == parent:
            children = sucs.values()[parent]
        else:
            for i in range(len(sucs)):
             if sucs.keys()[i] == parent:
               children = sucs.values()[i]
    else:
        for i in range(len(sucs)):
            if sucs.keys()[i] == parent:
               children = sucs.values()[i]
    
    return children

def loss(features,coords,scaler,w, data, predecs, children,img_nr, only_single):
    
    G, levels, y_p, x, boxes, ground_truths, alphas = data
    parent_child = 0.0
    layers = 0.0
    preds = np.dot(w,np.array(x).T)
    previous_layers = {}
    if only_single != -1:
        nodes = [only_single]
        y_selection = y_p[only_single]
        x_selection = x[only_single]
    else:
        nodes = list(G.nodes())[1:]
        y_selection = y_p
        x_selection = x
    for node in nodes:
        parent = predecs.values()[node-1]
        parent_pred = np.dot(w,x[parent])
        child_pred = np.dot(w,x[node])
        parent_child += (child_pred - parent_pred) if child_pred > parent_pred else 0
        if alphas[3] != 0:
            if parent in previous_layers.keys():
                children_layer = previous_layers[parent]
            else:
                children_layer,_,_ = count_per_level(features,coords,scaler,w,preds,img_nr, boxes, children[parent], '',[])
                previous_layers[parent] = children_layer
        else:
            children_layer = 0
        if squared_hinge_loss:
            layers += ((children_layer - parent_pred)**2) if children_layer > parent_pred else 0
        else:
            layers += (children_layer - parent_pred) if children_layer > parent_pred else 0
    ret = alphas[0] * ((y_selection - np.dot(w,np.array(x_selection).T)) ** 2).sum() + alphas[1] * np.dot(w,w) + alphas[2] * parent_child + alphas[3] * layers
    return ret

def learn_root(w,x,y,learning_rate,alphas):
    inner_prod = 0.0
    for f_ in range(len(w)):
        inner_prod += (w[f_] * x[f_])
    dloss = inner_prod - y
    for f_ in range(len(w)):
        w[f_] += (learning_rate * ((-x[f_] * dloss) + alphas[1] * w[f_]))
    return w

    
def like_scikit(features,coords,scaler,w,x,y,node,predecs,children,boxes,learning_rate,alphas,img_nr):
    x_node = x[node]
    y_node = y[node]
    inner_prod = 0.0
    for f_ in range(len(w)):
        inner_prod += (w[f_] * x_node[f_])
    dloss = inner_prod - y_node
    assert np.dot(w,x_node) == inner_prod
    #dloss = np.dot(w,x_node) - y_node
    parent = predecs.values()[node-1]
    parent_pred = np.dot(w,x[parent])
    child_pred = np.dot(w,x[node])
    preds = np.dot(w,np.array(x).T)
    children_layer = 0
    function1 = []
    for f_ in range(len(w)):
        parent_child = 0
        parent_pred = np.dot(w,x[parent])
        child_pred = np.dot(w,x[node])
        if child_pred > parent_pred:
            parent_child = (-x_node[f_] + x[parent][f_])
        layers_cons = 0
        if alphas[3] == 0:
            children_layer = 0
        else:
            children_layer, _, function1 = count_per_level(features,coords,scaler,w,preds,img_nr, boxes, children[parent], '',function1)
            #print 'lsk ', f_, children_layer, parent_pred, boxes[node][1], boxes[parent][1], learn_second_constraint
            if children_layer > parent_pred:
                if squared_hinge_loss:
                    layers_cons,function1 = train_per_level_b(children_layer,parent_pred,features,coords,scaler, x,node,img_nr, boxes, children[parent], parent, f_, function1)
                else:
                    layers_cons,function1 = train_per_level_a(features,coords,scaler, x,node,img_nr, boxes, children[parent], parent, f_, function1)
                #assert function1 == function2 #tested
        w[f_] += (learning_rate * (alphas[0]*(-x_node[f_] * dloss) + alphas[1] * w[f_] + alphas[2] * parent_child + alphas[3] * layers_cons))
    return w
    

def gradient(w,x,y,node,predecs,children,boxes,alphas,img_nr,f_):
    x_node = x[node]
    y_node = y[node]
    dloss = np.dot(w,x_node) - y_node
    parent = predecs.values()[node-1]
    parent_pred = np.dot(w,x[parent])
    child_pred = np.dot(w,x[node])
    preds = np.dot(w,np.array(x).T)
    if alphas[3] == 0:
        children_layer = 0
    else:
        children_layer = count_per_level(w,preds,img_nr, boxes, children[parent])
    parent_child = 0
    if child_pred > parent_pred:
        parent_child = x_node[f_]
    layers_cons = 0
    if children_layer > parent_pred:
        layers_cons = x_node[f_]
    return ((-x_node[f_] * dloss) + alphas[0] * w[f_] + alphas[1] * parent_child + alphas[2] * layers_cons)


def minibatch_(clf,scaler,w, loss__,mse,hinge1,hinge2,full_image,alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance, mode):
    if mode == 'loss_test' or mode == 'loss_scikit_test' or mode == 'levels_test':
        X_p, y_p, inv = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + 1, 'test', c)                
    else:
        X_p, y_p, inv = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + 1, 'training', c)        
    if X_p != []:
        boxes = []
        ground_truth = inv[0][2]
        img_nr = inv[0][0]
        if less_features:
            X_p = [fts[0:features_used] for fts in X_p]
        if os.path.isfile('/home/stahl/Coords_prop_windows/'+ (format(img_nr, "06d")) +'.txt'):
            f = open('/home/stahl/Coords_prop_windows/'+ (format(img_nr, "06d")) +'.txt', 'r')
        else:
            print 'warning'
        for line, y in zip(f, inv):
            tmp = line.split(',')
            coord = []
            for s in tmp:
                coord.append(float(s))
            boxes.append([coord, y[2]])
        assert(len(boxes)<500)
        boxes, y_p, X_p = sort_boxes(boxes, y_p, X_p, 0,500)
        
        if os.path.isfile('/home/stahl/GroundTruth/sheep_coords_for_features/'+ (format(img_nr, "06d")) +'.txt'):
            gr = open('/home/stahl/GroundTruth/sheep_coords_for_features/'+ (format(img_nr, "06d")) +'.txt', 'r')
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
            
        if mode == 'mean_variance':
            sum_x += np.array(pruned_x).sum(axis=0)
            n_samples += len(pruned_x)
            sum_sq_x +=  (np.array(pruned_x)**2).sum(axis=0)
            scaler.partial_fit(pruned_x)  # Don't cheat - fit only on training data
            return sum_x,n_samples,sum_sq_x, scaler
            
        # create_tree
        G, levels = create_tree(pruned_boxes)
        
        coords = []
        features = []
        if os.path.isfile('/home/stahl/Features_prop_windows/Features_upper/sheep'+ (format(img_nr, "06d")) +'.txt'):
            f = open('/home/stahl/Features_prop_windows/Features_upper/sheep'+ (format(img_nr, "06d")) +'.txt', 'r') 
        if os.path.isfile('/home/stahl/Features_prop_windows/upper_levels/sheep'+ (format(img_nr, "06d")) +'.txt'):
            f_c = open('/home/stahl/Features_prop_windows/upper_levels/sheep'+ (format(img_nr, "06d")) +'.txt', 'r+')
        for i,line in enumerate(f_c):
            str_ = line.rstrip('\n').split(',')
            cc = []
            for s in str_:
               cc.append(float(s))
            coords.append(cc)
        for i,line in enumerate(f):
            str_ = line.rstrip('\n').split(',')  
            ff = []
            for s in str_:
               ff.append(float(s))
            features.append(ff)
        if less_features:
            features = [fts[0:features_used] for fts in features]
        features = scaler.transform(features)
                
        assert len(coords) == len(features)
        
        # append x,y of intersections
        if learn_intersections:
            for inters,coord in zip(features,coords):
#                if inters not in pruned_x:
                pruned_x.append(inters)
                ol = 0.0
                ol = get_intersection_count(coord, ground_truths)
                pruned_y.append(ol)
        #normalize
        norm_x = []
        if normalize:
#            for p_x in pruned_x:
#                norm_x.append((p_x-mean)/variance)
            norm_x = scaler.transform(pruned_x)
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
    
            nodes = list(G.nodes())
            for node in nodes:
                print node
                if node == 0:
                    if alphas[0] != 0:
                        w = learn_root(w,norm_x[0],pruned_y[0],learning_rate,alphas)
                    else:
                        print 'learn nothing'
                else:
                    w = like_scikit(features,coords,scaler,w,norm_x,pruned_y,node,predecs,children,pruned_boxes,learning_rate,alphas,img_nr)
            return w, len(pruned_y)
        elif mode == 'scikit_train':
            clf.partial_fit(norm_x,pruned_y)
            return clf
        elif mode == 'loss_train' or mode == 'loss_test':
            loss__.append(loss(features,coords,scaler, w, data, predecs, children,img_nr,-1))
            mse.append(((data[2] - np.dot(w,np.array(data[3]).T)) ** 2).sum())
            a2 = alphas[2]
            data = (G, levels, pruned_y, norm_x, pruned_boxes, ground_truths, [0,0,a2,0])
            hinge1.append(loss(features,coords,scaler, w, data, predecs, children,img_nr,-1))
            a3 = alphas[3]
            data = (G, levels, pruned_y, norm_x, pruned_boxes, ground_truths, [0,0,0,a3])
            hinge2.append(loss(features,coords,scaler, w, data, predecs, children,img_nr,-1))
            full_image.append([pruned_y[0],np.dot(w,np.array(norm_x[0]).T)])
            return loss__, mse,hinge1,hinge2,full_image
        elif mode == 'loss_scikit_test' or mode == 'loss_scikit_train':
            loss__.append(((clf.predict(norm_x) - pruned_y)**2).sum())
            return loss__ 
        elif mode == 'finite_differences':
            feature = random.sample(range(4096),1)[0]
            #1. Pick an example z.
            example = random.sample(range(len(norm_x[1:])),1)[0]
            #2. Compute the loss Q(z, w) for the current w.
            Q = loss(features,coords,scaler, w,data, predecs, children,img_nr,example)
            #3. Compute the gradient g = ?w Q(z, w).
            g = gradient(w,norm_x,pruned_y,example,predecs,children,boxes,alphas,img_nr,feature)
            #4. Apply a slight perturbation w0 = w +d. For instance, change a single weight
            #by a small increment, or use d = -?g with ? small enough.
            w0 = w
            w0[feature] = w0[feature] + delta
            #5. Compute the new loss Q(z, w0
            #) and verify that Q(z, w0)  Q(z, w) + dg
            # Q(z, w + delta*e_i)  ( Q(z, w) + delta * g_i )
            Q_ = loss(features,coords,scaler, w0,data, predecs, children,img_nr,example)
            #print Q,Q_,g
            #print abs(Q_ - Q+(delta*g)) < 0.001
            #raw_input()
        elif mode == 'levels_train' or mode == 'levels_test':
            im = mpimg.imread('/home/stahl/Images/'+ (format(img_nr, "06d")) +'.jpg')
            plt.imshow(im)
            preds = []
            for i,x_ in enumerate(norm_x):
                preds.append(np.dot(w, x_))
            cpls = []
            truelvls = []
            used_boxes_ = []
            # to get prediction min and max for colorbar
            min_pred = 10
            max_pred = -5
            for level in levels:
                cpl,used_boxes,_ = count_per_level(features,coords,scaler,w, preds, img_nr, pruned_boxes,levels[level], '',[])
                if used_boxes is not None:
                    used_b_preds = [x[1] for x in used_boxes]
                    if used_b_preds != []:
                        if min(used_b_preds) < min_pred:
                            min_pred = min(used_b_preds)
                        if max(used_b_preds) > max_pred:
                            max_pred = max(used_b_preds)
            if min(preds) < min_pred:
                min_pred = min(preds)
            if max(preds) > max_pred:
                max_pred = max(preds)
            print'minmax of intersections: ', min_pred, max_pred
            cNorm  = colors.Normalize(vmin=min_pred, vmax=max_pred)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=pl.cm.jet)
            scalarMap.set_array(range(int(round(min_pred - 0.5)), int(round(max_pred + 0.5))))
            for pr_box, pr in zip(pruned_boxes,preds):
                pru_box = pr_box[0]
                colorVal = scalarMap.to_rgba(pr)
                ax = plt.gca()
                ax.add_patch(Rectangle((int(pru_box[0]), int(pru_box[1])), int(pru_box[2] - pru_box[0]), int(pru_box[3] - pru_box[1]), alpha=0.1, facecolor = colorVal, edgecolor = 'black'))
            for level in levels:
                #tru and truelvls was in order to check if count_per_level method is correct
                cpl,used_boxes,_ = count_per_level(features,coords,scaler,w, preds, img_nr, pruned_boxes,levels[level], '',[])
                #tru = count_per_level(None, pruned_y, img_nr, pruned_boxes,levels[level], 'gt')
                cpls.append(cpl)
                #plot image and predictions as color - only for debugging/testing
                if used_boxes is not None:
                    for u_box in used_boxes:
                        pru_box = pr_box[0]
                        colorVal = scalarMap.to_rgba(u_box[1])
                        #print u_box[0],u_box[1]
                        ax = plt.gca()
                        ax.add_patch(Rectangle((int(pru_box[0]), int(pru_box[1])), int(pru_box[2] - pru_box[0]), int(pru_box[3] - pru_box[1]), alpha=0.1, facecolor = colorVal, edgecolor = 'black'))
                #truelvls.append(tru)
            #print 'truth: ', pruned_y[0]
            matplotlib.pylab.colorbar(scalarMap, shrink=0.9)
            plt.draw()
            #plt.savefig('/home/stahl/'+str(img_nr))
            plt.clf()
            return cpls, truelvls, used_boxes_,pruned_boxes,preds
            
        
            
def main():
    test_imgs, train_imgs = get_seperation()
    # learn
#    if os.path.isfile('/home/stahl/Models/'+class_+c+'normalized_constrained.pickle'):
#        with open('/home/stahl/Models/'+class_+c+'normalized_constrained.pickle', 'rb') as handle:
#            w = pickle.load(handle)
#    else:
    weights = {}
    losses = {}
    gamma = 0.1
    epochs = 15
    images = 75
    subsamples = 20
    weights_visualization = {}
    learning_rates = [math.pow(10,-4)]
    learning_rates_ = {}
    if less_features:
        weights_sample = random.sample(range(features_used), 2)
    else:
        weights_sample = random.sample(range(4096), 10)
    all_alphas = [1]
    all_alphas_ = [1]
    n_samples = 0.0
    if less_features:
        sum_x = np.zeros(features_used)
        sum_sq_x = np.zeros(features_used)
    else:
        sum_x = np.zeros(4096)
        sum_sq_x = np.zeros(4096)

    plt.figure()
    mean = []
    variance = []
    scaler = []
    shuffled = range(0,images)
    clf = linear_model.SGDRegressor(learning_rate='constant', alpha = 0, eta0=learning_rates[0], shuffle=False, penalty='None',fit_intercept=False, n_iter=1)
    #normalize
    if normalize:
        scaler = MinMaxScaler()
        for minibatch in range(0,images):
            sum_x,n_samples,sum_sq_x,scaler = minibatch_(None,scaler,[], [],[],[],[],[],[],[],test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,None,None,'mean_variance')
        mean = sum_x/n_samples
        variance = (sum_sq_x - (sum_x * sum_x) / n_samples) / (n_samples)
    
    #check gradients using finite differences check
#    for minibatch in shuffled:
        #w = np.random.rand(4096)
#        w = np.zeros(4096)
#        differences = minibatch_(None,scaler,w, [],[],[],[],[],[0,0,0,0],[],test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance,'finite_differences')
        
    for alpha2 in all_alphas:
        for alpha3 in all_alphas_:
            for learning_rate0 in learning_rates:
                learning_rate = learning_rate0
                alphas = [1,0,1,1]
                if less_features:
                    w = 0.01 * np.random.rand(features_used)
                else:
                    w = 0.01 * np.random.rand(4096)                    
                loss_train = []
                loss_test = []
                loss_scikit_train = []
                loss_scikit_test = []
                mse_test = []
                mse_train = []
                hinge1_train = []
                hinge1_test = []
                hinge2_train = []
                hinge2_test = []
                full_image_test = []
                full_image_train = []
                learning_rate_again = []
                t = 0
                start = time.time()
                for epoch in range(epochs):
                    print epoch, learning_rate, alphas
                    if learning_rate0 in learning_rates_:
                        learning_rates_[learning_rate0].append(learning_rate)
                    else:
                        learning_rates_[learning_rate0] = [learning_rate]
                    #shuffle images, not boxes!
                    random.shuffle(shuffled)
                    for minibatch in shuffled:
                        w,le = minibatch_(None,scaler,w, [],[],[],[],[],alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance,'train')
                        t += le
                        clf = minibatch_(clf,scaler, [],[],[],[],[],[],alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance,'scikit_train')
                        print minibatch,', # boxes: ', t
                    #update learning_rate
                    learning_rate = learning_rate0 * (1+learning_rate0*gamma*t)**-1
                    learning_rate_again.append(learning_rate)
                    #compute average loss on training set
                    loss__train = []
                    loss__scikit_train = []
                    mse__train = []
                    hinge1__train = []
                    hinge2__train = []
                    full_image__train = []
                    for minibatch in range(0,images):
                        loss__train,mse__train,hinge1__train,hinge2__train,full_image__train = minibatch_(None,scaler,w, loss__train,mse__train,hinge1__train,hinge2__train,full_image__train,alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance,'loss_train')
                        loss__scikit_train = minibatch_(clf,scaler, loss__scikit_train,[],[],[],[],[],alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance,'loss_scikit_train')
                    #compute average loss on test set
                    loss__test = []
                    loss__scikit_test = []
                    mse__test = []
                    hinge1__test = []
                    hinge2__test = []
                    full_image__test = []
                    for minibatch in range(0,images):
                        loss__test,mse__test,hinge1__test,hinge2__test,full_image__test = minibatch_(None,scaler,w, loss__test,mse__test,hinge1__test,hinge2__test,full_image__test,alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance,'loss_test')
                        loss__scikit_test = minibatch_(clf,scaler, loss__scikit_test,[],[],[],[],[],alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance,'loss_scikit_test')
                    # save avg loss for plotting
                    temp_label = [alphas[0],alphas[1],alphas[2],alphas[3], learning_rate0]
                    loss_train.append(sum(loss__train)/len(loss__train))
                    loss_test.append(sum(loss__test)/len(loss__test))
                    loss_scikit_train.append(sum(loss__scikit_train)/len(loss__scikit_train))
                    loss_scikit_test.append(sum(loss__scikit_test)/len(loss__scikit_test))
                    mse_train.append(sum(mse__train)/len(mse__train))
                    mse_test.append(sum(mse__test)/len(mse__test))
                    hinge1_train.append(sum(hinge1__train)/len(hinge1__train))
                    hinge1_test.append(sum(hinge1__test)/len(hinge1__test))
                    hinge2_train.append(sum(hinge2__train)/len(hinge2__train))
                    hinge2_test.append(sum(hinge2__test)/len(hinge2__test))
                    full_image_train.append(((np.array([z[0] for z in full_image__train]) - np.array( [z_[1] for z_ in full_image__train]))**2).sum() / len(full_image__train))
                    full_image_test.append(((np.array([z[0] for z in full_image__test]) -np.array( [z_[1] for z_ in full_image__test]))**2).sum() / len(full_image__test))
                    print 'Loss: (train, test, mse_train, mse_test): %s %s %s %s' %(sum(loss__train)/len(loss__train), sum(loss__test)/len(loss__test), sum(mse__train)/len(mse__train), sum(mse__test)/len(mse__test))
                    print 'Loss: (hinge2 train) %s' % str(sum(hinge2__train)/len(hinge2__train))
                    print 'Loss: (hinge2 test) %s' % str(sum(hinge2__test)/len(hinge2__test))
                    # save sample weights for plotting
                    ww_ = []
                    for w_ in weights_sample:
                        ww_.append(w[w_])
                    if tuple(temp_label) in weights_visualization:
                        weights_visualization[alphas[0],alphas[1],alphas[2],alphas[3],learning_rate0].append(ww_)
                    else:
                        weights_visualization[alphas[0],alphas[1],alphas[2],alphas[3],learning_rate0] = [ww_]
                    
                    #get final prediction for images
                    #for minbatc in images
                    #tree
                    #levels
                    #max(iep(level)) == first level?
                    #regressor:mean?
                
                #save final weights
                end = time.time()
                print 'Time was :', (end - start)
                weights[alphas[0],alphas[1],alphas[2],alphas[3], learning_rate0] = w
                losses[alphas[0],alphas[1],alphas[2],alphas[3],learning_rate0] = loss_test[-1]
                #plot
                plt.ylabel('Loss')
                plt.xlabel('Iterations')
                #plt.plot(loss_test, '-r|',label='loss test set')
                #plt.plot(loss_train,'-r.', label='loss train set')
                #plt.plot(loss_scikit_train,'-c.', label='scikit train set')
                #plt.plot(loss_scikit_test,'-c|', label='scikit test set')
                #plt.plot(mse_train,'-g.', label='mse train set')
                #plt.plot(mse_test,'-g|', label='mse test set')
                #plt.plot(hinge1_train,'-y.', label='hinge1 train set')
                #plt.plot(hinge1_test,'-y|', label='hinge1 test set')
                plt.plot(hinge2_train,'-b.', label='hinge2 train set')
                plt.plot(hinge2_test,'-b|', label='hinge2 test set')
                #plt.plot(full_image_train,'-m.', label='full image train set')
                #plt.plot(full_image_test,'-m|', label='full image test set')
                plt.title('%s,%s,%s,%s,%s'%(learning_rate0,alphas[0], alphas[1],alphas[2],alphas[3]))
                plt.legend( loc='upper left', numpoints = 1, prop={'size':8})
                plt.savefig('/home/stahl/all_images%s_%s_%s_%s_%s_%s_%s_%s_b.png'%(learning_rate0, alphas[0],alphas[1],alphas[2],alphas[3],learn_intersections,subsampling,squared_hinge_loss))
                plt.clf()
                
                max_level_score_train = []
                max_level_score_test = []
                
                #plot tree and levels
                print 'levels: pred, true'
                for minibatch in shuffled:
                    cpls,trew,used_boxes_train,pruned_boxes_train,preds_train = minibatch_(clf,scaler,w, [],[],[],[],[],alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance, 'levels_train')         
                    print cpls,trew
                    max_level_score_train.append(max(cpls))
                    cpls,trew,used_boxes_test,pruned_boxes_test,preds_test = minibatch_(clf,scaler,w, [],[],[],[],[],alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance, 'levels_test')         
                    print cpls,trew
                    max_level_score_test.append(max(cpls))
                
                
                print len(full_image__train)
                mse_train_ = ((np.array([z[0] for z in full_image__train]) -np.array( [z_[1] for z_ in full_image__train]))**2).sum() / len(full_image__train)
                mse_test_ = ((np.array([z[0] for z in full_image__test]) - np.array([z_[1] for z_ in full_image__test]))**2).sum() / len(full_image__test)
                mse_mxlvl_train = ((np.array([z[0] for z in full_image__train]) -np.array(max_level_score_train ))**2).sum() / len(full_image__train)
                mse_mxlvl_test = ((np.array([z[0] for z in full_image__test]) - np.array(max_level_score_test))**2).sum() / len(full_image__test)
                
                plt.xlabel('ground truth')
                plt.ylabel('predictions')
                plt.plot([z[0] for z in full_image__train], [z_[1] for z_ in full_image__train],'mD', label='full image train')
                plt.plot([z[0] for z in full_image__test], [z_[1] for z_ in full_image__test],'bo',label='full image test')
                plt.plot([z[0] for z in full_image__train], max_level_score_train,'mp', label='max level train')
                plt.plot([z[0] for z in full_image__test], max_level_score_test,'bH',label='max level test')
                plt.plot([0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7],'r-')
                plt.xticks(np.arange(0.8,max(max([z[0] for z in full_image__test]),max([z[0] for z in full_image__train])+0.3)))
                plt.legend( loc='upper left', numpoints = 1 , prop={'size':8})
                plt.text(0.8,1.0,'MSE train:\n' + str(round(mse_train_,2)) + '\nMSE test:\n' + str(round(mse_test_,2))+'MSE mxlvl train:\n' + str(round(mse_mxlvl_train,2)) + '\nMSE mxlvl test:\n' + str(round(mse_mxlvl_test,2)), verticalalignment='bottom',
                             horizontalalignment='left',
                             fontsize=8,
                             bbox={'facecolor':'white', 'alpha':0.6, 'pad':10})
                plt.title('Predictions full image,%s,%s,%s,%s,%s'%(learning_rate0,alphas[0], alphas[1],alphas[2],alphas[3]))
                plt.savefig('/home/stahl/full_image_%s_%s_%s_%s_%s_%s_%s_%s_b.png'%(learning_rate0, alphas[0],alphas[1],alphas[2],alphas[3],learn_intersections,subsampling,squared_hinge_loss))
                plt.clf()
        
                print "model learned with learning rate: %s, alphas: %s, subsampling: %s, learn_intersections: %s, squared hinge loss 2nd constraint: %s "%(learning_rate0,alphas,subsampling,learn_intersections, squared_hinge_loss)
                with open('/home/stahl/Models/'+class_+c+'%s_%s_%s_%s_%s_%s_%s_%s_b.pickle'%(learning_rate0, alphas[0],alphas[1],alphas[2],alphas[3],learn_intersections,subsampling,squared_hinge_loss), 'wb') as handle:
                    pickle.dump(w, handle)
                with open('/home/stahl/Models/'+class_+c+'%s_%s_%s_%s_%s_%s_%s_%s_b_scaler.pickle'%(learning_rate0, alphas[0],alphas[1],alphas[2],alphas[3],learn_intersections,subsampling,squared_hinge_loss), 'wb') as handle:
                    pickle.dump(scaler, handle)

def create_tree(boxes):
    G = nx.Graph()
    levels = {}
    levels[0] = [0]
    G.add_node(0)
    if len(boxes) != 1:
        for box, i in zip(boxes[1:len(boxes)], range(1,len(boxes))):
            if (box[0][2]-box[0][0]) * (box[0][3]-box[0][1]) == 0: # some boxes have a surface area of 0 like (0,76,100,76)
                print box
                print 'surface area of box == 0', i
                continue
            possible_parents = []
            for box_, ii in zip(boxes, range(len(boxes))):
                if get_overlap_ratio.get_overlap_ratio(box[0], box_[0]) == 1 and box != box_:
                    possible_parents.append(ii)
                    #print i, '-', ii
            I = boxes[i][0]
            put_here = []
            for pp in possible_parents:
                p_h = True
                level = nx.shortest_path_length(G,0,pp)+1
                if level in levels:
                    for window in levels[level]:
                        II = boxes[window][0]
                        if get_overlap_ratio.get_overlap_ratio(I, II) == 1:
                            p_h = False
                    if p_h == True:
                        put_here.append(pp)
                else:
                    put_here.append(pp)
            parent = min(put_here)
            level = nx.shortest_path_length(G,0,parent)+1
            if level in levels:
                if parent not in levels[level]:
                    levels[level].append(i)
                G.add_edge(i,parent)
            else:
                levels[level] = [i]
                G.add_edge(i,parent)

    return G, levels



def count_per_level____(boxes, ground_truths, boxes_level): #previous version, keep
    if len(boxes_level) == 1:
        return boxes[boxes_level[0]][1]
    count_per_level = 0
    level_boxes = []
    index = {}
    nbrs = {}
#    print boxes_level, len(boxes)
    for i in boxes_level:
#        print i
        level_boxes.append(boxes[i][0])
        
    combinations = list(itertools.combinations(boxes_level, 2)) 
    G = nx.Graph()
    
    G.add_edges_from(combinations)
    
    for comb in combinations:
        set_ = []
        for c in comb:
            set_.append(boxes[c][0])
        I = get_set_intersection(set_)
        if I == []:
            G.remove_edges_from([comb])
    
    for u in G:
        index[u] = len(index)
        # Neighbors of u that appear after u in the iteration order of G.
        nbrs[u] = {v for v in G[u] if v not in index}

    queue = deque(([u], sorted(nbrs[u], key=index.__getitem__)) for u in G)
    # Loop invariants:
    # 1. len(base) is nondecreasing.
    # 2. (base + cnbrs) is sorted with respect to the iteration order of G.
    # 3. cnbrs is a set of common neighbors of nodes in base.
    while queue:
        base, cnbrs = map(list, queue.popleft())
        I = [0,0,1000,1000]
        for c in base:
            I = get_intersection(boxes[c][0], I)
        if len(base)%2==1:
            count_per_level += get_intersection_count(I, ground_truths)
        elif len(base)%2==0:
            count_per_level -= get_intersection_count(I, ground_truths)
                
        for i, u in enumerate(cnbrs):
            # Use generators to reduce memory consumption.
            queue.append((chain(base, [u]),
                          filter(nbrs[u].__contains__,
                                 islice(cnbrs, i + 1, None))))     
            
    return count_per_level


def get_set_intersection(set_):
    intersection = get_intersection(set_[0], set_[1])
    if intersection == []:
       return []
    for s in set_[2:len(set_)]:
        intersection = get_intersection(intersection, s)
        if intersection == []:
            return []
    return intersection

    
def test(w,x):
    return np.dot(x.reshape(1,len(x)),np.transpose(w))
    
    
def sort_boxes(boxes, preds, X_p, from_, to):
    sorted_boxes = []
    sorted_preds = []
    sorted_features = []
    decorated = [((box[0][3]-box[0][1])*(box[0][2]-box[0][0]), i) for i, box in enumerate(boxes)]
    decorated.sort()
    for box, i in reversed(decorated):
        sorted_boxes.append(boxes[i])
        sorted_preds.append(preds[i])
        sorted_features.append(X_p[i])
    return sorted_boxes[from_:to], sorted_preds[from_:to], sorted_features[from_:to]
                                

def get_seperation():
    file = open('/home/stahl/Generic counting/IO/test.txt')
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
        file = open('/home/stahl/Generic counting/IO/ClassImages/'+ class_+'.txt', 'r')
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

# read boxes for coords -> window size

        if os.path.isfile('/home/stahl/Coords_prop_windows/'+ (format(i, "06d")) +'.txt'):
            f = open('/home/stahl/Coords_prop_windows/'+ (format(i, "06d")) +'.txt', 'r')
        else:
            print 'warning'
        boxes = []
        for line in f:
            tmp = line.split(',')
            coord = []
            for s in tmp:
                coord.append(float(s))
            boxes.append(coord)
        if fs != []:
            trained = True
            if baseline == 1 or baseline == 2:                    
                features.append(fs)
                tmp = get_labels(i, criteria)
                ll = int(tmp[0])
                labels.append(ll)
                investigate.append([i, 0, ll])
                #if phase == 'test':
                #    im = mpim.imread('/home/t/Schreibtisch/Thesis/VOCdevkit1/VOC2007/JPEGImages/'+ (format(i, "06d")) +'.jpg')/home/stahl/Images
                #    plt.imshow(im)
            else:
                features.extend(fs)
                l = get_labels(i, criteria)
                labels.extend(l)
                for ind in range(len(l)):
                    surface_area = (boxes[ind][3] - boxes[ind][1]) * (boxes[ind][2] - boxes[ind][0])
                    investigate.append([i, ind, l[ind], surface_area])
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
    if os.path.isfile('/home/stahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt'):
        file = open('/home/stahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt', 'r')
    else:
        print 'warning /home/stahl/Features_prop_windows/SS_Boxes'+ (format(i, "06d")) +'.txt does not exist '
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
    if os.path.isfile('/home/stahl/Coords_prop_windows/Labels/'+(format(i, "06d")) + '_' + class_ + '_' + criteria + '.txt'):
        file = open('/home/stahl/Coords_prop_windows/Labels/'+(format(i, "06d")) + '_' + class_ + '_' + criteria + '.txt', 'r')
    else:
        print 'warning /home/stahl/Coords_prop_windows/Labels'+(format(i, "06d")) + '_' + class_ + '_' + criteria + '.txt does not exist '
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
        
def count_per_level(features,coords,scaler,w,preds,img, boxes, boxes_level, mode, function):
    #tested
    sums = np.zeros(len(boxes_level))
    if len(boxes_level) == 1:
        return preds[boxes_level[0]],None,[]
    if function != []:
        count_per_level = 0
        for fun in function:
            if 'p' in fun[1]:
                if '+' in fun[0]:
                    count_per_level += preds[fun[2]]
                elif '-' in fun[0]:
                    count_per_level -= preds[fun[2]]
                else:
                    print 'wrong symbol 0', fun[0]
            elif 'f' in fun[1]:
                if '+' in fun[0]:
                    count_per_level += np.dot(w,features[fun[2]])
                elif '-' in fun[0]:
                    count_per_level -= np.dot(w,features[fun[2]])
                else:
                    print 'wrong symbol 0', fun[0]
            else:
                print 'wrong symbol 1', fun[1]
        return count_per_level, [], function
    else:
        used_boxes = []
        level_boxes = []
        for i in boxes_level:
            level_boxes.append(boxes[i][0])
        combinations = list(itertools.combinations(boxes_level, 2)) 
        G = nx.Graph()
        G.add_edges_from(combinations)
        for comb in combinations:
            set_ = []
            for c in comb:
                set_.append(boxes[c][0])
            I = get_set_intersection(set_)
            if I == []:
                G.remove_edges_from([comb])
        sums,used_boxes,function = sums_of_all_cliques(features,coords,scaler,w, G, preds, boxes, sums, img, mode)
    return iep(sums),used_boxes,function
    
    
def iep(sums):
    ret = 0
    for summe, ij in zip(sums, range(len(sums))):
        if ij % 2 == 0:
            ret += summe
        else:
            ret -= summe
    return ret
    
    
def sums_of_all_cliques(features, coords, scaler,w, G, preds, boxes, sums, img_nr, mode):
    feat_exist = True #must be false in order to write
    real_b = [b[0] for b in boxes]
    used_boxes = []
    write_coords = []
    length = 1
    index = {}
    nbrs = {}
    function = []
    if mode == 'gt':
        if os.path.isfile('/home/stahl/GroundTruth/sheep_coords_for_features/'+ (format(img_nr, "06d")) +'.txt'):
            gr = open('/home/stahl/GroundTruth/sheep_coords_for_features/'+ (format(img_nr, "06d")) +'.txt', 'r')
        ground_truths = []
        for line in gr:
           tmp = line.split(',')
           ground_truth = []
           for s in tmp:
              ground_truth.append(int(s))
           ground_truths.append(ground_truth)
    for u in G:
        index[u] = len(index)
        # Neighbors of u that appear after u in the iteration order of G.
        nbrs[u] = {v for v in G[u] if v not in index}

    queue = deque(([u], sorted(nbrs[u], key=index.__getitem__)) for u in G)
    # Loop invariants:
    # 1. len(base) is nondecreasing.
    # 2. (base + cnbrs) is sorted with respect to the iteration order of G.
    # 3. cnbrs is a set of common neighbors of nodes in base.
    while queue:
        base, cnbrs = map(list, queue.popleft())
        if len(base) > length:
            length = len(base)
        I = [0,0,1000,1000]
        for c in base:
            if I != []:
               I = get_intersection(boxes[c][0], I)
        if I != []:
          if I in real_b:
             ind = real_b.index(I)
             sums[len(base)-1] += preds[ind]
             if len(base)%2==1:
                  function.append(['+','p',ind])
             elif len(base)%2==0:
                  function.append(['-','p',ind])
             else:
                  print ''
             #used_boxes.append([I,np.dot(w,features[ind])])
          else:
             if feat_exist == True:
                if mode != 'gt':
                    if I in coords and I != []:
                      ind = coords.index(I)
                      sums[len(base)-1] += np.dot(w,features[ind])
                      used_boxes.append([I,np.dot(w,features[ind])])
                      if len(base)%2==1:
                          function.append(['+','f',ind])
                      elif len(base)%2==0:
                          function.append(['-','f',ind])
                      else:
                          print ''
                    else:
                        print 'not found', I
                elif mode == 'gt':
                    sums[len(base)-1] += get_intersection_count(I, ground_truths)
                   
             else:
                if I not in coords and I not in write_coords:
                     write_coords.append(I)
        for i, u in enumerate(cnbrs):
            # Use generators to reduce memory consumption.
            queue.append((chain(base, [u]),
                          filter(nbrs[u].__contains__,
                                 islice(cnbrs, i + 1, None))))
                                 
    return sums, used_boxes, function
    
    
#def train_per_level(features,coords,scaler, x,node,img_nr, boxes, children, parent, feat):
#    if len(children) == 1:
#        return (-x[node][feat] + x[parent][feat])
#    children_boxes = []
#    for i in children:
#        children_boxes.append(boxes[i][0])
#        
#    # create graph G from combinations possible        
#    combinations = list(itertools.combinations(children, 2)) 
#    G = nx.Graph()
#    G.add_edges_from(combinations)
#    for comb in combinations:
#        set_ = []
#        for c in comb:
#            set_.append(boxes[c][0])
#        I = get_set_intersection(set_)
#        if I == []:
#            G.remove_edges_from([comb])
#    
#    feat_exist = True #must be false in order to write
#    real_b = [b[0] for b in boxes]
#    write_coords = []
#    length = 1
#    index = {}
#    nbrs = {}
#
#    for u in G:
#        index[u] = len(index)
#        # Neighbors of u that appear after u in the iteration order of G.
#        nbrs[u] = {v for v in G[u] if v not in index}
#
#    queue = deque(([u], sorted(nbrs[u], key=index.__getitem__)) for u in G)
#    # Loop invariants:
#    # 1. len(base) is nondecreasing.
#    # 2. (base + cnbrs) is sorted with respect to the iteration order of G.
#    # 3. cnbrs is a set of common neighbors of nodes in base.
#    count_per_level = 0
#    while queue:
#        base, cnbrs = map(list, queue.popleft())
#        if len(base) > length:
#            length = len(base)
#        I = [0,0,1000,1000]
#        for c in base:
#            if I != []:
#               I = get_intersection(boxes[c][0], I)
#        if I != []:
#          if I in real_b:
#             ind = real_b.index(I)
#             if len(base)%2==1:
#                 count_per_level += x[ind][feat]
#             else:
#                 count_per_level -= x[ind][feat]
#          else:
#             if feat_exist == True:
#                if I in coords and I != []:
#                  ind = coords.index(I)
#                  if len(base)%2==1:
#                     count_per_level += features[ind][feat]
#                  else:
#                     count_per_level -= features[ind][feat]
#                else:
#                    print 'not found', I
#                   
#             else:
#                if I not in coords and I not in write_coords:
#                     write_coords.append(I)
#        for i, u in enumerate(cnbrs):
#            # Use generators to reduce memory consumption.
#            queue.append((chain(base, [u]),
#                          filter(nbrs[u].__contains__,
#                                 islice(cnbrs, i + 1, None))))
#
#    if feat_exist == False:
#       print 'write coords', len(write_coords)
#       for coor in write_coords:
#          f_c.seek(0,2)
#          f_c.write(str(coor[0])+','+str(coor[1])+','+str(coor[2])+','+str(coor[3]))
#          f_c.write('\n')
#    return (- count_per_level + x[parent][feat])

def train_per_level_b(children_layer,parent_pred,features,coords,scaler, x,node,img_nr, boxes, children, parent, feat, function):
    if len(children) == 1:
        return (children_layer - parent_pred) *(-x[node][feat] + x[parent][feat]), []
    count_per_level = 0
    if function != []:
        for fun in function:
            if 'p' in fun[1]:
                if '+' in fun[0]:
                    count_per_level += x[fun[2]][feat]
                elif '-' in fun[0]:
                    count_per_level -= x[fun[2]][feat]
                else:
                    print 'wrong symbol', fun[0]
            elif 'f' in fun[1]:
                if '+' in fun[0]:
                    count_per_level += features[fun[2]][feat]
                elif '-' in fun[0]:
                    count_per_level -= features[fun[2]][feat]
                else:
                    print 'wrong symbol 0', fun[0]
            else:
                print 'wrong symbol 1', fun[1]
        return (children_layer - parent_pred) *(- count_per_level + x[parent][feat]), function
    else:
        children_boxes = []
        for i in children:
            children_boxes.append(boxes[i][0])
            
        # create graph G from combinations possible        
        combinations = list(itertools.combinations(children, 2)) 
        G = nx.Graph()
        G.add_edges_from(combinations)
        for comb in combinations:
            set_ = []
            for c in comb:
                set_.append(boxes[c][0])
            I = get_set_intersection(set_)
            if I == []:
                G.remove_edges_from([comb])
        
        feat_exist = True #must be false in order to write
        real_b = [b[0] for b in boxes]
        write_coords = []
        length = 1
        index = {}
        nbrs = {}
    
        for u in G:
            index[u] = len(index)
            # Neighbors of u that appear after u in the iteration order of G.
            nbrs[u] = {v for v in G[u] if v not in index}
    
        queue = deque(([u], sorted(nbrs[u], key=index.__getitem__)) for u in G)
        # Loop invariants:
        # 1. len(base) is nondecreasing.
        # 2. (base + cnbrs) is sorted with respect to the iteration order of G.
        # 3. cnbrs is a set of common neighbors of nodes in base.
        while queue:
            base, cnbrs = map(list, queue.popleft())
            if len(base) > length:
                length = len(base)
            I = [0,0,1000,1000]
            for c in base:
                if I != []:
                   I = get_intersection(boxes[c][0], I)
            if I != []:
              if I in real_b:
                 ind = real_b.index(I)
                 if len(base)%2==1:
                     count_per_level += x[ind][feat]
                     function.append(['+','p',ind])
                 else:
                     count_per_level -= x[ind][feat]
                     function.append(['-','p',ind])
              else:
                 if feat_exist == True:
                    if I in coords and I != []:
                      ind = coords.index(I)
                      if len(base)%2==1:
                         count_per_level += features[ind][feat]
                         function.append(['+','f',ind])
                      else:
                         count_per_level -= features[ind][feat]
                         function.append(['-','f',ind])
                    else:
                        print 'not found', I
                       
                 else:
                    if I not in coords and I not in write_coords:
                         write_coords.append(I)
            for i, u in enumerate(cnbrs):
                # Use generators to reduce memory consumption.
                queue.append((chain(base, [u]),
                              filter(nbrs[u].__contains__,
                                     islice(cnbrs, i + 1, None))))
        return (children_layer - parent_pred) * (- count_per_level + x[parent][feat]), function
    
    
def train_per_level_a(features,coords,scaler, x,node,img_nr, boxes, children, parent, feat, function):
    if len(children) == 1:
        return (-x[node][feat] + x[parent][feat]), []
    count_per_level = 0
    if function != []:
        for fun in function:
            if 'p' in fun[1]:
                if '+' in fun[0]:
                    count_per_level += x[fun[2]][feat]
                elif '-' in fun[0]:
                    count_per_level -= x[fun[2]][feat]
                else:
                    print 'wrong symbol', fun[0]
            elif 'f' in fun[1]:
                if '+' in fun[0]:
                    count_per_level += features[fun[2]][feat]
                elif '-' in fun[0]:
                    count_per_level -= features[fun[2]][feat]
                else:
                    print 'wrong symbol 0', fun[0]
            else:
                print 'wrong symbol 1', fun[1]
        return (- count_per_level + x[parent][feat]), function
    else:
        children_boxes = []
        for i in children:
            children_boxes.append(boxes[i][0])
            
        # create graph G from combinations possible        
        combinations = list(itertools.combinations(children, 2)) 
        G = nx.Graph()
        G.add_edges_from(combinations)
        for comb in combinations:
            set_ = []
            for c in comb:
                set_.append(boxes[c][0])
            I = get_set_intersection(set_)
            if I == []:
                G.remove_edges_from([comb])
        
        feat_exist = True #must be false in order to write
        real_b = [b[0] for b in boxes]
        write_coords = []
        length = 1
        index = {}
        nbrs = {}
    
        for u in G:
            index[u] = len(index)
            # Neighbors of u that appear after u in the iteration order of G.
            nbrs[u] = {v for v in G[u] if v not in index}
    
        queue = deque(([u], sorted(nbrs[u], key=index.__getitem__)) for u in G)
        # Loop invariants:
        # 1. len(base) is nondecreasing.
        # 2. (base + cnbrs) is sorted with respect to the iteration order of G.
        # 3. cnbrs is a set of common neighbors of nodes in base.
        while queue:
            base, cnbrs = map(list, queue.popleft())
            if len(base) > length:
                length = len(base)
            I = [0,0,1000,1000]
            for c in base:
                if I != []:
                   I = get_intersection(boxes[c][0], I)
            if I != []:
              if I in real_b:
                 ind = real_b.index(I)
                 if len(base)%2==1:
                     count_per_level += x[ind][feat]
                     function.append(['+','p',ind])
                 else:
                     count_per_level -= x[ind][feat]
                     function.append(['-','p',ind])
              else:
                 if feat_exist == True:
                    if I in coords and I != []:
                      ind = coords.index(I)
                      if len(base)%2==1:
                         count_per_level += features[ind][feat]
                         function.append(['+','f',ind])
                      else:
                         count_per_level -= features[ind][feat]
                         function.append(['-','f',ind])
                    else:
                        print 'not found', I
                       
                 else:
                    if I not in coords and I not in write_coords:
                         write_coords.append(I)
            for i, u in enumerate(cnbrs):
                # Use generators to reduce memory consumption.
                queue.append((chain(base, [u]),
                              filter(nbrs[u].__contains__,
                                     islice(cnbrs, i + 1, None))))
        return (- count_per_level + x[parent][feat]), function
    
if __name__ == "__main__":
#    cProfile.run('main()')
    main()
