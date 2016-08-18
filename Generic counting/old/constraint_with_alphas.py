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
from itertools import chain, islice
import itertools
from get_intersection import get_intersection
import cProfile
from collections import deque
from get_intersection_count import get_intersection_count
import matplotlib.colors as colors
#from load import get_seperation, get_data,get_image_numbers
import matplotlib.image as mpimg
from utils import create_tree, find_children, sort_boxes, surface_area, extract_coords
from ml import count_per_level, constrained_regression, loss, learn_root
import time
import matplotlib.cm as cmx
from scipy import optimize
import time
from utils import get_set_intersection
import cProfile
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from matplotlib.patches import Rectangle

#class_ = 'car'
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
prune_tree_levels = 3
jans_idea = True

def get_used_proposals(G, boxes, coords, levels):
    used_ind = {'prop':[],'inters':[]}
    children_boxes = []
    for level in levels:
        for i in levels[level]:
            children_boxes.append(boxes[i][0])
            
        # create graph G from combinations possible        
        combinations = list(itertools.combinations(levels[level], 2)) 
        G = nx.Graph()
        G.add_edges_from(combinations)
        for comb in combinations:
            set_ = []
            for c in comb:
                set_.append(boxes[c][0])
            I = get_set_intersection(set_)
            if I == []:
                G.remove_edges_from([comb])
        
        real_b = [b[0] for b in boxes]
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
                 if ind not in used_ind['prop']:
                     used_ind['prop'].append(ind)
              else:
                if I in coords and I != []:
                  ind = coords.index(I)
                  if ind not in used_ind['inters']:
                     used_ind['inters'].append(ind)
                else:
                    print 'not found', I
            for i, u in enumerate(cnbrs):
                # Use generators to reduce memory consumption.
                queue.append((chain(base, [u]),
                              filter(nbrs[u].__contains__,
                                     islice(cnbrs, i + 1, None))))
    return used_ind


def minibatch_(all_train_imgs,all_test_imgs,clf,scaler,w, loss__,mse,hinge1,hinge2,full_image,alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance, mode,mous):
    if mode == 'loss_test' or mode == 'loss_scikit_test' or mode == 'levels_test':
        X_p, y_p, inv = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + 1, 'test', c, subsamples)
    else:
        X_p, y_p, inv = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + 1, 'training', c, subsamples)        
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
            boxes.append([coord,y[2]])
        assert(len(boxes)<1500)
        boxes, y_p, X_p = sort_boxes(boxes, y_p, X_p, 0,1500)
        
        gr = []
        if os.path.isfile('/home/stahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d"))):
            gr = open('/home/stahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d")), 'r')
        ground_truths = []
        if gr != []: #if no class image -> no ground truth. (I think this is only needed for learn _ntersection)
            for line in gr:
               tmp = line.split(',')
               ground_truth = []
               for s in tmp:
                  ground_truth.append(int(s))
               ground_truths.append(ground_truth)
        
        if mode == 'mean_variance':
            scaler.partial_fit(X_p)  # Don't cheat - fit only on training data
            return scaler
            
        # create_tree
        G, levels = create_tree(boxes)
        
        #prune tree to only have levels which fully cover the image
        # tested
        if prune_fully_covered:
            nr_levels_covered = 100
            total_size = surface_area(boxes, levels[0])
            for level in levels:
                sa = surface_area(boxes, levels[level])
                sa_co = sa/total_size
                if sa_co != 1.0:
                    G.remove_nodes_from(levels[level])
                else:
                    nr_levels_covered = level
            levels = {k: levels[k] for k in range(0,nr_levels_covered + 1)}
        
        #either subsampling or prune_fully_covered
        #assert(subsampling != prune_fully_covered)
        
        # prune levels, speedup + performance 
        levels = {k:v for k,v in levels.iteritems() if k<prune_tree_levels}
        
        #prune G in order to remove nodes of the lower levels
        remaining_nodes = []
        for lev in levels.values():
            remaining_nodes.extend(lev)
        for g_node in G.nodes():
            if g_node not in remaining_nodes:
                G.remove_node(g_node)
        
        coords = []
        features = []
        f_c = []
        f = []
        
        if learn_intersections and not prune_fully_covered:
            if os.path.isfile('/home/stahl/Features_prop_windows/upper_levels/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples)):
                f_c = open('/home/stahl/Features_prop_windows/upper_levels/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples), 'r+')
            else:
                if mode == 'extract_train' or mode == 'extract_test':                
                    print 'coords for %s with %s samples have to be extracted'%(img_nr,subsamples)
                    f_c = open('/home/stahl/Features_prop_windows/upper_levels/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples), 'w')
                    for level in levels:
                        levl_boxes = extract_coords(levels[level], boxes)
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
                    #print "EuVisual /home/stahl/Images/%s.jpg /home/stahl/Features_prop_windows/Features_upper/sheep_%s_%s.txt --eudata /home/koelma/EuDataBig --imageroifile /home/stahl/Features_prop_windows/upper_levels/sheep_%s_%s.txt"%((format(img_nr, "06d")),format(img_nr, "06d"),subsamples,format(img_nr, "06d"),subsamples)
                    os.system("EuVisual /home/stahl/Images/%s.jpg /home/stahl/Features_prop_windows/Features_upper/%s_%s_%s.txt --eudata /home/koelma/EuDataBig --imageroifile /home/stahl/Features_prop_windows/upper_levels/%s_%s_%s.txt"%(class_,(format(img_nr, "06d")),format(img_nr, "06d"),subsamples,class_,format(img_nr, "06d"),subsamples))
                    if os.path.isfile('/home/stahl/Features_prop_windows/upper_levels/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples)):
                        f_c = open('/home/stahl/Features_prop_windows/upper_levels/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples), 'r')
                    else:
                        f_c = []
            coords = []
                
            if os.path.isfile('/home/stahl/Features_prop_windows/Features_upper/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples)):
                f = open('/home/stahl/Features_prop_windows/Features_upper/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples), 'r') 
                
                
        elif prune_fully_covered:
            if os.path.isfile('/home/stahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d"))):
                f_c = open('/home/stahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d")), 'r+')
                
                
            else:
                if mode == 'extract_train' or mode == 'extract_test':                
                    print 'coords for %s with fully_cover_tree samples have to be extracted'%(img_nr)
                    f_c = open('/home/stahl/Features_prop_windows/upper_levels/%s_%s_fully_cover_tree.txt'%(class_,format(img_nr, "06d")), 'w')
                    for level in levels:
                        levl_boxes = extract_coords(levels[level], boxes)
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
                    #print "EuVisual /home/stahl/Images/%s.jpg /home/stahl/Features_prop_windows/Features_upper/sheep_%s_%s.txt --eudata /home/koelma/EuDataBig --imageroifile /home/stahl/Features_prop_windows/upper_levels/sheep_%s_%s.txt"%((format(img_nr, "06d")),format(img_nr, "06d"),subsamples,format(img_nr, "06d"),subsamples)
                    os.system("EuVisual /home/stahl/Images/%s.jpg /home/stahl/Features_prop_windows/Features_upper/%s_%s_fully_cover_tree.txt --eudata /home/koelma/EuDataBig --imageroifile /home/stahl/Features_prop_windows/upper_levels/%s_%s_fully_cover_tree.txt"%(class_,(format(img_nr, "06d")),format(img_nr, "06d"),class_,format(img_nr, "06d")))
                    if os.path.isfile('/home/stahl/Features_prop_windows/upper_levels/%s_%s_fully_cover_tree.txt'%(class_,format(img_nr, "06d"))):
                        f_c = open('/home/stahl/Features_prop_windows/upper_levels/%s_%s_fully_cover_tree.txt'%(class_,format(img_nr, "06d")), 'r')
                    else:
                        f_c = []
            coords = []
                
            if os.path.isfile('/home/stahl/Features_prop_windows/Features_upper/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d"))):
                f = open('/home/stahl/Features_prop_windows/Features_upper/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d")), 'r') 
                        
                
#        else: #we don't need to load intersections
#            if os.path.isfile('/home/stahl/Features_prop_windows/Features_upper/%s%s.txt'%(class_,format(img_nr, "06d"))):
#                f = open('/home/stahl/Features_prop_windows/Features_upper/%s%s.txt'%(class_,format(img_nr, "06d")), 'r') 
#            if os.path.isfile('/home/stahl/Features_prop_windows/upper_levels/%s%s.txt'%(class_,format(img_nr, "06d"))):
#                f_c = open('/home/stahl/Features_prop_windows/upper_levels/%s%s.txt'%(class_,format(img_nr, "06d")), 'r+')
#            else:
#                print '/home/stahl/Features_prop_windows/upper_levels/%s%s.txt does not exist'%(class_,format(img_nr, "06d"))
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
        if normalize and features != []:
            features = scaler.transform(features)
        
        print len(y_p), len(X_p)
        print len(features), len(coords)
        assert len(coords) == len(features)
        
        # append x,y of intersections
        #if learn_intersections:
        #    for inters,coord in zip(features,coords):
#                if inters not in pruned_x:
        #        X_p.append(inters)
        #        ol = 0.0
        #        ol = get_intersection_count(coord, ground_truths)
        #        y_p.append(ol)
        print len(y_p), len(X_p)
        #normalize
        norm_x = []
        if normalize:
#            for p_x in pruned_x:
#                norm_x.append((p_x-mean)/variance)
            norm_x = scaler.transform(X_p)
        else:
            norm_x = X_p
        data = (G, levels, y_p, norm_x, boxes, ground_truths, alphas)
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
            if alphas[2] == 0 and alphas[3] == 0: #just learn proposals and intersections
                # only use proposals and intersections used in pruned tree
                used_ind = get_used_proposals(G, boxes, coords, levels)
                used_x = []
                used_y = []
                for ind in used_ind['prop']:
                    used_x.append(norm_x[ind])
                    used_y.append(y_p[ind])
                for ind in used_ind['inters']:
                    used_x.append(features[ind])
                    used_y.append(get_intersection_count(coords[ind], ground_truths))
                print len(used_x),len(used_y)
                for x_i,y_i in zip(used_x,used_y):
                    w = learn_root(w,x_i,y_i,learning_rate,alphas)
            else:
                nodes = list(G.nodes())
                for node in nodes:
                    if node == 0:
                        if alphas[0] != 0:
                            w = learn_root(w,norm_x[0],y_p[0],learning_rate,alphas)
                        else:
                            print 'learn nothing'
                    else:
                        w = constrained_regression(class_,features,coords,scaler,w,norm_x,y_p,node,predecs,children,boxes,learning_rate,alphas,img_nr, squared_hinge_loss)
            return w, len(y_p), len(G.nodes())
        elif mode == 'scikit_train':
            print norm_x,y_p
            clf.partial_fit(norm_x,y_p)
            return clf
        elif mode == 'loss_train' or mode == 'loss_test':
            loss__.append(loss(class_,squared_hinge_loss,features,coords,scaler, w, data, predecs, children,img_nr,-1))
#            mse.append(((data[2] - np.dot(w,np.array(data[3]).T)) ** 2).sum())
#            a2 = alphas[2]
#            data = (G, levels, y_p, norm_x, boxes, ground_truths, [0,0,a2,0])
#            hinge1.append(loss(class_,squared_hinge_loss,features,coords,scaler, w, data, predecs, children,img_nr,-1))
#            a3 = alphas[3]
#            data = (G, levels, y_p, norm_x, boxes, ground_truths, [0,0,0,a3])
#            hinge2.append(loss(class_,squared_hinge_loss,features,coords,scaler, w, data, predecs, children,img_nr,-1))
            full_image.append([y_p[0],max(0,np.dot(w,np.array(norm_x[0]).T))])
            return loss__, full_image
        elif mode == 'loss_scikit_test' or mode == 'loss_scikit_train':
            loss__.append(((clf.predict(norm_x) - y_p)**2).sum())
            return loss__ 
        elif mode == 'levels_train' or mode == 'levels_test':
            #im = mpimg.imread('/home/stahl/Images/'+ (format(img_nr, "06d")) +'.jpg')
            preds = []
            for i,x_ in enumerate(norm_x):
                preds.append(np.dot(w, x_))
            cpls = []
            truelvls = []
            used_boxes_ = []
            total_size = surface_area(boxes, levels[0])
            fully_covered_score = 0.0
            fully_covered_score_lvls = 0.0
            covered_levels = []
            for level in levels:
                #tru and truelvls was in order to check if count_per_level method is correct
                cpl,used_boxes,_ = count_per_level(class_,features,coords,scaler,w, preds, img_nr, boxes,levels[level], '',[])
                cpl = max(0,cpl)
                if used_boxes != []:
                    used_boxes_.append(used_boxes[0][1])
                tru,_,_ = count_per_level(class_,features,coords,scaler,w, preds, img_nr, boxes,levels[level], 'gt',[])
                cpls.append(cpl)
                sa = surface_area(boxes, levels[level])
                sa_co = sa/total_size
                if sa_co == 1.0:
                   fully_covered_score += cpl
                   fully_covered_score_lvls += 1
                   covered_levels.append(cpl)
                truelvls.append(tru)
            return cpls, truelvls, used_boxes_,boxes,preds,fully_covered_score/fully_covered_score_lvls,covered_levels
            
        
            
def main():
    all_test_imgs,all_train_imgs = get_seperation()
    # learn
#    if os.path.isfile('/home/stahl/Models/'+class_+c+'normalized_constrained.pickle'):
#        with open('/home/stahl/Models/'+class_+c+'normalized_constrained.pickle', 'rb') as handle:
#            w = pickle.load(handle)
#    else:
    weights = {}
    losses = {}
    gamma = 0.1
    epochs = 1
    test_imgs = all_test_imgs
    new = True
    regs = [1e-4]
    classes = ['bus','tvmonitor','bird','dog','cat','aeroplane','motorbike','boat','bottle','pottedplant','sheep','cow','horse','bicycle','train','car','chair','diningtable','sofa','person']
    
    mous = 'whole'
    for cur_class in classes:
        global class_
        class_ = cur_class
        
        if mous != 'whole':
            train_imgs = get_image_numbers(all_test_imgs,all_train_imgs,class_)
        else:
            train_imgs = all_train_imgs
        if subsampling:
            subsamples = 1
        else:
            subsamples = 10000
        weights_visualization = {}
        #train_imgs = 
        learning_rates = [math.pow(10,-4)]
        learning_rates_ = {}
        if less_features:
            weights_sample = random.sample(range(features_used), 2)
        else:
            weights_sample = random.sample(range(4096), 10)
        all_alphas = [1]
        all_alphas_ = [1]
        n_samples = 0.0
        
        train_imgs = train_imgs[0:5]
        test_imgs = test_imgs[0:5]
        
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
        shuffled = range(0,len(train_imgs))
        converged = 100

        #normalize
        if normalize:
            scaler = MinMaxScaler()
            for minibatch in range(0,len(train_imgs)):
                print 'normalizing', minibatch
                scaler = minibatch_(all_train_imgs,all_test_imgs,None,scaler,[], [],[],[],[],[],[],[],test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,None,None,'mean_variance',mous)

        for alpha3 in all_alphas_:
            for learning_rate0 in learning_rates:
                learning_rate = learning_rate0
                alphas = [1,regs[0],0,0]
                if less_features:
                    w = 0.01 * np.random.rand(features_used)
                else:
                    w = 0.01 * np.random.rand(4096)                    
                learning_rate_again = []
                t = 0
                for epoch in range(epochs):
                    print epoch, learning_rate, alphas, class_
                    if learning_rate0 in learning_rates_:
                        learning_rates_[learning_rate0].append(learning_rate)
                    else:
                        learning_rates_[learning_rate0] = [learning_rate]
                    #shuffle images, not boxes!
                    random.shuffle(shuffled)
                    for it,minibatch in enumerate(shuffled):
                        print 'training', it
                        w,le, nr_nodes = minibatch_(all_train_imgs,all_test_imgs,None,scaler,w, [],[],[],[],[],alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance,'train',mous)
                        t += le
                        #clf = minibatch_(clf,scaler, [],[],[],[],[],[],alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance,'scikit_train')
                        print 'training ', it,'(',len(shuffled),epoch,')',', # boxes: ', t, ' len(tree) ', nr_nodes
                        #update learning_rate
                        learning_rate = learning_rate0 * (1+learning_rate0*gamma*t)**-1
                    learning_rate_again.append(learning_rate)
                
                max_level_score_train = []
                max_level_score_test = []
                max_level_score_train_nn = []
                max_level_score_test_nn = []
                
                full_cover_level_score_train = []
                full_cover_level_score_test = []
                full_cover_level_score_train_nn = []
                full_cover_level_score_test_nn = []
                
                level_mse_train = []
                level_mse_test = []
            
                min_intersection_train = 100
                min_intersection_test = 100
                
                covered_levels_train = []
                covered_levels_test = []
                
                full_image__train = []
                full_image__test = []
                
                avg_gt_train = []
                avg_gt_test = []
                loss__train = []
                
                for minibatch in range(0,len(train_imgs)):
                    loss__train, full_image__train = minibatch_(all_train_imgs,all_test_imgs,clf,scaler,w, loss__,mse,hinge1,hinge2,full_image,alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance,'loss_train',mous)
                # check for convergence
                new_loss = loss__train.sum()
                print new_loss
                if new_loss < converged:
                    if converged - new_loss < 0.01:
                        break
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
            full_image__test = []
            median_test = []
            median_test_nn = []
            min_level_test = []
            min_level_test_nn = []
            just_level = {}
            just_level_nn = {}
            clte = []
            lmte = []

            for minibatch in range(0,len(test_imgs)):
                _,full_image__test = minibatch_(all_train_imgs,all_test_imgs,[],scaler,w, [],[],[],[],full_image__test,alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance, 'loss_test',mous)
                cpls,trew,used_boxes_test,pruned_boxes_test,preds_test,fully_covered_score,covered_test = minibatch_(all_train_imgs,all_test_imgs,[],scaler,w, [],[],[],[],[],alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance, 'levels_test',mous)
                max_level_score_test.append(max(cpls))                        
                full_cover_level_score_test.append(fully_covered_score)
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
                    full_cover_level_score_test_nn.append(fully_covered_score)
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
                covered_levels_test = np.hstack((covered_levels_test,((np.array(covered_test)-np.array(trew[0:len(covered_test)]))**2)))
            

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
    
                
#                    #assert cpls == covered_test
#                    for i,le in enumerate(zip(cpls,trew)):
#                        plt.plot(le[1],le[0],'ro',label='level %s'%(i))
#                    plt.plot(range(8),range(8),'k-')
#                    plt.ylabel('Prediction')
#                    plt.xlabel('Ground Truth')
#                    plt.title('Prediction per level')
#                    plt.savefig('/home/stahl/Levels.png')
#                    plt.clf()
#                   
#                    #save final weights
#                    end = time.time()
#                    print 'Time was :', (end - start)
#                    #plot
#                    plt.ylabel('Loss')
#                    plt.xlabel('Iterations')
#                    plt.plot(loss_test, '-r|',label='loss test')
#                    plt.plot(loss_train,'-r.', label='loss train')
#                    plt.plot(full_image_train,'-m.', label='full image train')
#                    plt.plot(full_image_test,'-m|', label='full image test')
#                    plt.plot(mse_mxlvl_train, '-b.',label='max level train')
#                    plt.plot(mse_mxlvl_test, '-b|',label='max level test')
#                    plt.plot(mse_fllycover_train, '-c.',label='fully cover levels train')
#                    plt.plot(mse_fllycover_test, '-c|',label='fully cover levels test')
#                    #plt.plot(cltr, '-y.', label='covered levels train')
#                    #plt.plot(clte, '-y|', label='covered levels test')
#                    plt.plot(lmtr, '-g.', label='levels train')
#                    plt.plot(lmte, '-g|', label='levels test')
#                    plt.title('%s,%s,%s'%(learning_rate0,alphas[0], alphas[1]))
#                    plt.legend( loc='upper left', numpoints = 1, prop={'size':8})
#                    if subsampling:
#                        plt.savefig('/home/stahl/all_images%s_%s_%s_%s_%s_%s_%s_%s_b.png'%(learning_rate0, alphas[0],alphas[1],learn_intersections,subsampling,squared_hinge_loss,subsamples,reg))
#                    elif prune_fully_covered:
#                        plt.savefig('/home/stahl/all_images%s_%s_%s_%s_%s_%s_%s_%s_%s.png'%(learning_rate0, alphas[0],alphas[1],learn_intersections,subsampling,squared_hinge_loss,prune_fully_covered,reg,mous))
#                    else:
#                        plt.savefig('/home/stahl/all_images%s_%s_%s_%s_%s_%s_%s_%s_b.png'%(learning_rate0, alphas[0],alphas[1],learn_intersections,subsampling,squared_hinge_loss,reg))
#                    plt.clf()
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
            plt.savefig('/home/stahl/constrained_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_clipped_more.png'%(class_,learning_rate0, alphas[0],alphas[3],learn_intersections,subsampling,squared_hinge_loss,prune_fully_covered,alphas[1],mous,prune_tree_levels,jans_idea,epochs))
            print "model learned with learning rate: %s, alphas: %s, subsampling: %s, learn_intersections: %s, squared hinge loss 2nd constraint: %s "%(learning_rate0,alphas,subsampling,learn_intersections, squared_hinge_loss)
            print new                    
            if new:                    
                with open('/home/stahl/Models/constrained_'+class_+c+'%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.pickle'%(learning_rate0, alphas[0],alphas[3],learn_intersections,subsampling,squared_hinge_loss,prune_fully_covered,reg,mous,prune_tree_levels,epochs), 'wb') as handle:
                    pickle.dump(w, handle)
                with open('/home/stahl/Models/constrained_'+class_+c+'%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_scaler.pickle'%(learning_rate0, alphas[0],alphas[3],learn_intersections,subsampling,squared_hinge_loss,prune_fully_covered,alphas[1],mous,prune_tree_levels), 'wb') as handle:
                    pickle.dump(scaler, handle)        
            plt.clf()

def get_image_numbers(test_imgs, train_imgs,class_):
    file = open('/home/stahl/Generic counting/IO/ClassImages/'+ class_+'.txt', 'r')
    training_images = []
    test_images = []
    for line in file:
        im_nr = int(line)
        if im_nr in train_imgs:
            training_images.append(im_nr)
        elif  im_nr in test_imgs:
            test_images.append(im_nr)
    im = []
    other = [x for x in train_imgs if x not in training_images]
    for t in range(len(training_images)):
        im.append(other[random.randint(0,len(other)-1)])
    #assert(len(im)==len(training_images))
    training_images.extend(im)
    return training_images
    
def get_seperation():
    file = open('/home/stahl/Generic counting/IO/test.txt')
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
        start = time.time()
        fs = get_features(i, subsamples)
#        print time.time() - start
#        start = time.time()
#        fs_np = get_features_il(i,subsamples)
#        print time.time() - start
#        start = time.time()
#        fs_np = get_features_np(i,subsamples)
#        print time.time() - start
#        start = time.time()
#        fs_pd = get_features_pd(i,subsamples)
#        print time.time() - start        
#        raw_input()

# read boxes for coords -> window size

        if os.path.isfile('/home/stahl/Coords_prop_windows/'+ (format(i, "06d")) +'.txt'):
            f = open('/home/stahl/Coords_prop_windows/'+ (format(i, "06d")) +'.txt', 'r')
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
    if os.path.isfile('/home/stahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt'):
        file = open('/home/stahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt', 'r')
    else:
        print 'warning /home/stahl/Features_prop_windows/SS_Boxes'+ (format(i, "06d")) +'.txt does not exist '
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
    
def get_features_np(i, subsamples):
    features = []
    if os.path.isfile('/home/stahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt'):
        pan = np.loadtxt('/home/stahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt',delimiter=',')        
    else:
        print 'warning /home/stahl/Features_prop_windows/SS_Boxes'+ (format(i, "06d")) +'.txt does not exist '
        return features
    return pan[0:subsamples]

    
def get_features_il(i, subsamples):
    features = []
    if os.path.isfile('/home/stahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt'):
        d = iter_loadtxt('/home/stahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt')
    else:
        print 'warning /home/stahl/Features_prop_windows/SS_Boxes'+ (format(i, "06d")) +'.txt does not exist '
        return features
    return d[0:subsamples]
    
def get_features_pd(i, subsamples):
    features = []
    if os.path.isfile('/home/stahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt'):
        d = pd.read_csv('/home/stahl/Features_prop_windows/SS_Boxes/'+ (format(i, "06d")) +'.txt')
    else:
        print 'warning /home/stahl/Features_prop_windows/SS_Boxes'+ (format(i, "06d")) +'.txt does not exist '
        return features
    return d[0:subsamples]

    
def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data


def get_labels(class_,i, criteria, subsamples):
    labels = []
    if os.path.isfile('/home/stahl/Coords_prop_windows/Labels/'+(format(i, "06d")) + '_' + class_ + '_' + criteria + '.txt'):
        file = open('/home/stahl/Coords_prop_windows/Labels/'+(format(i, "06d")) + '_' + class_ + '_' + criteria + '.txt', 'r')
    else:
        print 'warning /home/stahl/Coords_prop_windows/Labels'+(format(i, "06d")) + '_' + class_ + '_' + criteria + '.txt does not exist '
        return np.zeros(subsamples)
    for i_l, line in enumerate(file):
        tmp = line.split()[0]
        labels.append(float(tmp))
        if i_l == subsamples - 1:
            break
    return labels

def bool_rect_intersect(A, B):
    return not (B[0]>A[2] or B[2]<A[0] or B[3]<A[1] or B[1]>A[3]), 1/((A[2]- A[0] + 1)*(A[3]-A[1] + 1))
        #return !(r2.left > r1.right || r2.right < r1.left || r2.top > r1.bottom ||r2.bottom < r1.top);
    
if __name__ == "__main__":
#    cProfile.run('main()')
    main()
