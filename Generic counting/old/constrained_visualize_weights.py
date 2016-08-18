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
import networkx as nx
import pyximport; pyximport.install(pyimport = True)
import get_overlap_ratio
import itertools
from get_intersection import get_intersection
from collections import deque
from itertools import chain, islice
from get_intersection_count import get_intersection_count
from scipy import optimize
import time
import cProfile

class_ = 'sheep'
baseline = False
add_window_size = False
iterations = 0
subsampling = True
c = 'partial'

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

def loss(w, data, predecs, children):
    
    G, levels, y_p, x, boxes, ground_truths, alphas = data
    parent_child = 0.0
    layers = 0.0
    for node in list(G.nodes())[1:]:
        parent = predecs.values()[node-1]
        parent_pred = np.dot(w,x[parent])
        child_pred = np.dot(w,x[node])
        parent_child += child_pred - parent_pred if child_pred > parent_pred else 0
        children_layer = count_per_level(boxes, ground_truths, children[parent])
        layers += children_layer - parent_pred if children_layer > parent_pred else 0
    ret = sum((y_p - np.dot(w,np.array(x).reshape(4096,len(x)))) ** 2) + alphas[0] * np.dot(w.reshape(1,4096),w.reshape(4096,1)) + alphas[1] * parent_child + alphas[2] * layers
    first = sum((y_p - np.dot(w,np.array(x).reshape(4096,len(x)))) ** 2) + alphas[0] * np.dot(w.reshape(1,4096),w.reshape(4096,1))
    if alphas[1] == 0 and alphas[2] == 0 and alphas[0] != 0:
        assert ret == first, 'first != ret'
    return ret
   
def loss_simple(w,data):
    y_p = data[2]
    x = data[3]
    alpha1 = data[6][0]
    y_p = y_p[1:]
    x = x[1:]
    return sum((y_p - np.dot(w,np.array(x).reshape(4096,len(x)))) ** 2) + alpha1 * np.dot(w.reshape(1,4096),w.reshape(4096,1))

def loss_prime(w,data,predecs,children,node,feature):
    G, levels, y, x, boxes, ground_truths, alphas = data
    if len(x) <= node:
        print len(x), node
    x_f = x[node][feature]
    y_node = y[node]
    parent = predecs.values()[node-1]
    parent_pred = y[parent]
    child_pred = y[node]
    parent_child = 0
    if child_pred > parent_pred:
        parent_child = x_f
    #children = find_children(sucs, parent)
    children_layer = count_per_level(boxes, ground_truths, children[parent])
    layers = 0
    if children_layer > parent_pred:
        layers = x_f
    
    window_error = (y_node - w[feature] * x_f) *(x_f) 
    alpha_weights = alphas[0] * w[feature]
    first_constraint = alphas[1] * parent_child
    second_constrained = alphas[2] * layers * np.sign(x_f * w[feature])
    if alphas[1] == 0 and alphas[2] == 0:
        assert first_constraint == 0 and second_constrained == 0, "constraints not zero, despite alphas are"
    ret = window_error + alpha_weights + first_constraint + second_constrained
    return ret
    
def loss_prime_simple(w,node,feature,data):
    x = data[3]
    y = data[2]
    alpha1 = data[6][0]
    x_f = x[node][feature]    
    y_node = y[node]
    if feature == 10:
        print w[feature], x_f, w[feature] * x_f, (y_node - w[feature] * x_f) * (x_f)
    ret = (y_node - w[feature] * x_f) * (x_f)
    return ret + alpha1 * w[feature]

def update_weights(w,data,predecs,children,node, learning_rate):
    len_features = len(data[3][0])
    w_new = np.zeros(len_features)
    for feature_ in range(len_features):
        w_new[feature_] = loss_prime_simple(w,node,feature_,data)
    return w + learning_rate * w_new
    
def like_scikit(w,x,y, learning_rate,alphas):
    inner_product = 0.0    
    for f_ in range(len(x)):
        inner_product += (w[f_] * x[f_])
    dloss = inner_product - y
    update = dloss * -learning_rate
    for f_ in range(len(x)):
        w[f_] += ((x[f_] * update) + alphas[0] * w[f_])
    return w


def minibatch_(w, loss__,alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance, mode):
    X_p, y_p, inv = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + 1, 'training', c)                
    if X_p != []:
        boxes = []
        ground_truth = inv[0][2]
        img_nr = inv[0][0]
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
        for i, y_ in enumerate(y_p):
            if y_ > 0:
                pruned_x.append(X_p[i])
                pruned_y.append(y_p[i])
                pruned_boxes.append(boxes[i])
        
        if subsampling and pruned_boxes > subsamples:
            pruned_x = pruned_x[0:subsamples]
            pruned_y = pruned_y[0:subsamples]
            pruned_boxes = pruned_boxes[0:subsamples]
            
        if mode == 'mean_variance':
            sum_x += np.array(pruned_x).sum(axis=0)
            n_samples += len(pruned_x)
            sum_sq_x +=  (np.array(pruned_x)**2).sum(axis=0)
            return sum_x,n_samples,sum_sq_x
            
        # create_tree
        G, levels = create_tree(pruned_boxes)
        norm_x = []
        
        #normalize
        for p_x in pruned_x:
            norm_x.append((p_x-mean)/variance)
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
            nodes = list(G.nodes())[1:]
            for node in nodes:
                w = like_scikit(w,norm_x[node],pruned_y[node],learning_rate,alphas)
                #w = update_weights(w,data,predecs,children,node, learning_rate)
        else:
            loss__.append(loss_simple(w,data))#(w, data, predecs, children))
    if mode == 'train':
        return w, len(pruned_y)
    else:
        return loss__
            
def main():
    test_imgs, train_imgs = get_seperation()
    # learn
#    if os.path.isfile('/home/stahl/Models/'+class_+c+'normalized_constrained.pickle'):
#        with open('/home/stahl/Models/'+class_+c+'normalized_constrained.pickle', 'rb') as handle:
#            w = pickle.load(handle)
#    else:
    loss_ = {}
    weights = {}
    gamma = 0.5
    epochs = 50
    images = 10
    subsamples = 10
    weights_visualization = {}
    learning_rates = [math.pow(10,-3),math.pow(10,-4),math.pow(10,-5),math.pow(10,-6)]
    learning_rates_ = {}
    weights_sample = random.sample(range(4096), 10)
    all_alphas = [0,math.pow(10,-1),math.pow(10,-2),math.pow(10,-3)]
    sum_x = np.zeros(4096)
    n_samples = 0.0
    sum_sq_x = np.zeros(4096)
    #TODO: normalize
    for minibatch in range(0,images):
        sum_x,n_samples,sum_sq_x = minibatch_([], [],[],[],test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,None,None,'mean_variance')
    mean = sum_x/n_samples
    variance = (sum_sq_x - (sum_x * sum_x) / n_samples) / (n_samples - 1)
    print mean,variance
    raw_input()
    for alpha1 in all_alphas:
        for learning_rate0 in learning_rates:
            learning_rate = learning_rate0
            alphas = [alpha1,0,0]
            w = np.zeros(4096)
            change = -100000
            for epoch in range(epochs):
                print epoch, learning_rate
                if learning_rate0 in learning_rates_:
                    learning_rates_[learning_rate0].append(learning_rate)
                else:
                    learning_rates_[learning_rate0] = [learning_rate]
                #shuffle images, not boxes!
                shuffled = range(0,images)
                random.shuffle(shuffled)
                for minibatch in shuffled:
                    w,t = minibatch_(w, [],alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance,'train')
                            
                #update learning_rate
                learning_rate = learning_rate0 * (1+learning_rate0*gamma*t)**-1
                #compute average loss on training set
                loss__ = []
                for minibatch in range(0,images):
                    loss__ = minibatch_(w, loss__,alphas,learning_rate,test_imgs, train_imgs,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance,'test')
                # save avg loss for plotting
                temp_label = [alphas[0], learning_rate0]
                llloss = sum(loss__)/len(loss__)
                if tuple(temp_label) in loss_:
                    change = llloss - loss_[alphas[0], learning_rate0][-1]
                    print 'avg', llloss, ' change: ',  change
                if tuple(temp_label) in loss_:
                    loss_[alphas[0], learning_rate0].append(sum(loss__)/len(loss__))
                else:
                    loss_[alphas[0], learning_rate0] = [sum(loss__)/len(loss__)]                          
                          
                # save sample weights for plotting
                ww_ = []
                for w_ in weights_sample:
                    ww_.append(w[w_])
                temp_label = [alphas[0],learning_rate0]
                if tuple(temp_label) in weights_visualization:
                    weights_visualization[alphas[0],learning_rate0].append(ww_)
                else:
                    weights_visualization[alphas[0],learning_rate0] = [ww_]
                    
                #TODO: update learning rate
            
            
            
            #save final weights
            weights[alphas[0], learning_rate0] = w
                        
    final_model_losses = [x[-1] for x in loss_.values()]
    best_model_index = final_model_losses.index(min(final_model_losses))
    a1, learning_rate_0 = loss_.keys()[best_model_index]
    print a1, learning_rate_0
    w_best = weights[a1, learning_rate_0]
    #plot
    for i,l in zip(loss_.keys(),loss_.values()):
        to_plot = [math.log(a) for a in loss_[i]]
        labl = "alpha1=%s"%i[0]
        plt.plot(range(len(loss_[i])),to_plot,'-', label=labl)
        plt.title('Learning rate=%s'%learning_rate_0)
    plt.xlabel('Iterations')
    plt.ylabel('Log(Loss)')
    plt.legend()
    plt.savefig('/home/stahl/debugwithalphas.png')
    
    plt.figure()
    for l in learning_rates_:
        plt.plot(range(len(learning_rates_[l])),learning_rates_[l],label='start=%s'%l)
    plt.xlabel('Iterations')
    plt.ylabel('Learning rate')
    plt.legend()
    plt.savefig('/home/stahl/learning_ratewithalphas.png')
    
    plt.figure()
    for in_ in range(len(weights_sample)):
        refactor = [weights_visualization[a1,learning_rate_0][x][in_] for x in range(len(weights_visualization[a1,learning_rate_0]))]
        plt.plot(range(len(refactor)),refactor,'-')
    plt.xlabel('Iterations')
    plt.ylabel('Weights')
    plt.savefig('/home/stahl/best_weights_pruned_withalphas.png')
    print "model learned"
    with open('/home/stahl/Models/'+class_+c+'normalized_constrained_best.pickle', 'wb') as handle:
        pickle.dump(w_best, handle)
        
    sys.exit("Error message")
    #TODO: compute average loss test set using best configuration on hold out set
    loss__ = []
    for minibatch in range(0,100,1):
        print alphas, learning_rate, minibatch
        X_p, y_p, inv = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + 1, 'test', c)                
        if X_p != []:
            #TODO: prune?
            boxes = []
            ground_truth = inv[0][2]
            img_nr = inv[0][0]
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

            # create_tree
            G, levels = create_tree(boxes)
            # normalize
            new_matrix = preprocessing.normalize(X_p, norm='l2', axis=0)
            root = boxes[0]
            boxes_ = boxes[1:]
            for box, y,i_ in zip(boxes_, y_p[1:],range(1,len(boxes))):
                if i_ in G.nodes():
                   parent = nx.dfs_predecessors(G,i_).values().index(i_)
                   for i, b_i in enumerate(levels.values()):
                      if any(x == parent for x in b_i):
                          level = i
                          break
                else:
                   print len(new_matrix), i_
                   if i_ >= len(new_matrix):
                      new_matrix = new_matrix[0:-1]
                      y_p = y_p[0:-1]
                   else:
                      new_matrix = np.vstack((new_matrix[0:i_],new_matrix[i_+1:]))
                      y_p_temp = y_p[0:i_]
                      y_p_temp.append(y_p[i_+1:])
                      y_p = y_p_temp
  #              y_p = y_p[0:8]
  #              new_matrix = new_matrix[0:8]
            data = (G, levels, y_p, new_matrix, boxes, ground_truths, alphas)
            sucs = nx.dfs_successors(G)
            predecs = nx.dfs_predecessors(G)
            #preprocess: node - children
            children = {}
            last = -1
            #print sucs, predecs
            for node,children_ in zip(sucs.keys(),sucs.values()):
                #print node, children_, predecs.values()[node-1]
                #print node,children_, last+1
                #raw_input()
                if node != last+1:
                    for i in range(last+1,node):
                        children[i] = []
                    children[node] = children_
                elif node == last +1:
                    children[node] = children_
                last = node
            loss__.append(loss(w, data, predecs, children))

    temp_label = [alphas[0],alphas[1],alphas[2],learning_rate]
    if tuple(temp_label) in loss_:
        loss_[alphas[0],alphas[1],alphas[2],learning_rate].append(sum(loss__)/len(loss__))
    else:
        loss_[alphas[0],alphas[1],alphas[2],learning_rate] = [sum(loss__)/len(loss__)]
                

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



def count_per_level(boxes, ground_truths, boxes_level):
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
    
    
def count_per_level_pred(boxes, ground_truths, boxes_level):
    if len(boxes_level) == 1:
        return boxes[boxes_level[0]][1]
    count_per_level = 0
    level_boxes = []
    index = {}
    nbrs = {}
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
                #    im = mpim.imread('/home/t/Schreibtisch/Thesis/VOCdevkit1/VOC2007/JPEGImages/'+ (format(i, "06d")) +'.jpg')
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
        
    
    
if __name__ == "__main__":
#    cProfile.run('main()')
    main()
