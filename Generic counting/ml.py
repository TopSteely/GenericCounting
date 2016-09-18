import numpy as np
from utils import get_set_intersection, surface_area
import os
import networkx as nx
from collections import deque
from get_intersection_count import get_intersection_count
from get_intersection import get_intersection
import itertools
from itertools import chain, islice


def constrained_regression(class_,features,coords,scaler,w,x,y,node,predecs,children,boxes,learning_rate,alphas,img_nr,squared_hinge_loss):
    x_node = x[node]
    y_node = y[node]
    inner_prod = 0.0
    for f_ in range(len(w)):
        inner_prod += (w[f_] * x_node[f_])
    dloss = (inner_prod - y_node)
    #assert np.dot(w,x_node) == inner_prod
    #dloss = np.dot(w,x_node) - y_node
    parent = predecs[node]
    print 'p: ', parent
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
            children_layer, _, function1 = count_per_level([],class_,features,coords,scaler,w,preds,img_nr, boxes, children[parent], '',function1)
            #print 'lsk ', f_, children_layer, parent_pred, boxes[node][1], boxes[parent][1], learn_second_constraint
            if children_layer > parent_pred:
                if squared_hinge_loss:
                    layers_cons,function1 = train_per_level_b(children_layer,parent_pred,features,coords,scaler, x,node,img_nr, boxes, children[parent], parent, f_, function1)
                else:
                    layers_cons,function1 = train_per_level_a(features,coords,scaler, x,node,img_nr, boxes, children[parent], parent, f_, function1)
                #assert function1 == function2 #tested
        w[f_] += (learning_rate * (alphas[0]*(-x_node[f_] * dloss) + alphas[1] * w[f_] + alphas[2] * parent_child + alphas[3] * layers_cons))
        #w[f_] += (learning_rate * ( alphas[0]* ((-x_node[f_] * dloss))+ alphas[1] * w[f_] ))
    return w
    
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
        
    
def count_per_level(used_ind,class_,features,coords,scaler,w,preds,img, boxes, boxes_level, mode, function):
    #tested
    sums = np.zeros(len(boxes_level))
    if used_ind != []:
        new_preds = []
        ccc = 0
        for hhh in range(len(boxes)):
            if hhh in used_ind['prop']:
                new_preds.append(preds[ccc])
                ccc += 1
            else:
                new_preds.append(0)
        preds = new_preds
    gr = []
    if len(boxes_level) == 1:
        if mode == '':
            return preds[boxes_level[0]],[],[]
        elif mode == 'gt':
            return boxes[boxes_level[0]][1],[],[]
    if mode == 'gt':
        if os.path.isfile('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img, "06d"))):
            gr = open('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img, "06d")), 'r')
        ground_truths = []
        if gr != []:
            for line in gr:
               tmp = line.split(',')
               ground_truth = []
               for s in tmp:
                  ground_truth.append(int(s))
               ground_truths.append(ground_truth)
    if function != []:
        count_per_level = 0
        for fun in function:
            if 'p' in fun[1]:
                if '+' in fun[0]:
                    if mode == 'gt':
                        count_per_level += boxes[fun[2]][1]
                    else:
                        count_per_level += preds[fun[2]]
                elif '-' in fun[0]:
                    if mode == 'gt':
                        count_per_level -= boxes[fun[2]][1]
                    else:
                        count_per_level -= preds[fun[2]]
                else:
                    print 'wrong symbol 0', fun[0]
            elif 'f' in fun[1]:
                if '+' in fun[0]:
                    if mode == 'gt':
                        count_per_level += get_intersection_count(coords[fun[2]], ground_truths)
                    else:
                        count_per_level += np.dot(w,features[fun[2]])
                elif '-' in fun[0]:
                    if mode == 'gt':
                        count_per_level -= get_intersection_count(coords[fun[2]], ground_truths)
                    else:
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
        sums,used_boxes,function = sums_of_all_cliques(class_,features,coords,scaler,w, G, preds, boxes, sums, img, mode)
    return iep(sums),used_boxes,function
    
    
def iep(sums):
    ret = 0
    for summe, ij in zip(sums, range(len(sums))):
        if ij % 2 == 0:
            ret += summe
        else:
            ret -= summe
    return ret
    
def sums_of_all_cliques(class_,features, coords, scaler,w, G, preds, boxes, sums, img_nr, mode):
    feat_exist = True #must be false in order to write
    real_b = [b[0] for b in boxes]
    used_boxes = []
    write_coords = []
    length = 1
    index = {}
    nbrs = {}
    gr = []
    function = []
    if mode == 'gt':
        if os.path.isfile('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d"))):
            gr = open('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d")), 'r')
        ground_truths = []
        if gr != []:
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
             if mode == '':
                 sums[len(base)-1] += preds[ind]
             elif mode == 'gt':
                 sums[len(base)-1] += boxes[ind][1]
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
    
def loss(class_,squared_hinge_loss,features,coords,scaler,w, data, predecs, children,img_nr, only_single):
    
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
    if alphas[2] > 0 and alphas[3] > 0:
        for node in nodes:
            parent = predecs.values()[node-1]
            parent_pred = np.dot(w,x[parent])
            child_pred = np.dot(w,x[node])
            parent_child += (child_pred - parent_pred) if child_pred > parent_pred else 0
            if alphas[3] != 0:
                if parent in previous_layers.keys():
                    children_layer = previous_layers[parent]
                else:
                    children_layer,_,_ = count_per_level([],class_,features,coords,scaler,w,preds,img_nr, boxes, children[parent], '',[])
                    previous_layers[parent] = children_layer
            else:
                children_layer = 0
            if squared_hinge_loss:
                layers += ((children_layer - parent_pred)**2) if children_layer > parent_pred else 0
            else:
                layers += (children_layer - parent_pred) if children_layer > parent_pred else 0
    ret = alphas[0] * ((y_selection - np.dot(w,np.array(x_selection).T)) ** 2).sum() + alphas[1] * np.dot(w,w) + alphas[2] * parent_child + alphas[3] * layers
    #ret = alphas[0] * ((y_selection - np.dot(w,np.array(x_selection).T)) ** 2).sum() + alphas[1] * np.dot(w,w)
    return ret

def learn_root(w,x,y,learning_rate,alphas):
    inner_prod = 0.0
    for f_ in range(len(w)):
        inner_prod += (w[f_] * x[f_])
    #dloss = max(0,inner_prod) - y
    dloss = inner_prod - y
    for f_ in range(len(w)):
        w[f_] += (learning_rate * ((-x[f_] * dloss)))# + alphas[1] * w[f_]))
    return w
    
def tree_level_regression(class_,function,levels,level,features,coords,scaler,w,x,y,node,predecs,children,boxes,learning_rate,alphas,img_nr,jans_idea):
    x_node = np.zeros(len(w))
    dloss = 0
    if alphas[0] != 0:
        x_node = x[node]
        y_node = y[node]
        inner_prod = 0.0
        for f_ in range(len(w)):
            inner_prod += (w[f_] * x_node[f_])
        dloss = inner_prod - y_node
        #assert np.dot(w,x_node) == inner_prod
    #dloss = np.dot(w,x_node) - y_node
    preds = np.dot(w,np.array(x).T)
    for f_ in range(len(w)):
        if alphas[1] == 0:
            tree_lvls = 0
        else:
            tree_lvls,function = train_per_level_c(y[0],class_,w,preds,levels,level,features,coords,scaler, x,node,img_nr, boxes, f_, function,jans_idea)
        #print alphas[0],-x_node[f_] , dloss , alphas[1] , tree_lvls,alphas[2] ,w[f_]
        w[f_] += (learning_rate * (alphas[0]*(-x_node[f_] * dloss) + alphas[1] * tree_lvls + alphas[2] * w[f_]))
    return w,function
    
def tree_level_loss(class_,features,coords,scaler,w, data, predecs, children,img_nr, only_single, functions):
    
    G, levels, y_p, x, boxes, ground_truths, alphas = data
    preds = []
    for i,x_ in enumerate(x):
        preds.append(np.dot(w, x_))
    cpls = []
    truelvls = []
    used_boxes_ = []
    total_size = surface_area(boxes, levels[0])
    fully_covered_score = 0.0
    fully_covered_score_lvls = 0.0
    covered_levels = []
    for level in levels:
        if img_nr in functions:
            if level in functions[img_nr]:
                function = functions[img_nr][level]
            else:
                function = []
        else:
            functions[img_nr] = {}
            function = []
        cpl,used_boxes,function = count_per_level([],class_,features,coords,scaler,w, preds, img_nr, boxes,levels[level], '',function)
        if used_boxes != []:
            used_boxes_.append(used_boxes[0][1])
        if level not in functions[img_nr]:
            functions[img_nr][level] = function
        tru = y_p[0]
        cpls.append(cpl)
        sa = surface_area(boxes, levels[level])
        sa_co = sa/total_size
        if sa_co == 1.0:
           fully_covered_score += cpl
           fully_covered_score_lvls += 1
           covered_levels.append(cpl)
        truelvls.append(tru)
    ret = alphas[0] * ((y_p - np.dot(w,np.array(x).T)) ** 2).sum() + alphas[1] * max(0,((np.array(cpls)-np.array(truelvls))**2).sum() / len(cpls)) +  alphas[2] * np.dot(w,w)
    return ret
    
def train_per_level_c(y,class_,w,preds,levels,level,features,coords,scaler, x,node,img_nr, boxes, feat, function, jans_idea):
    tru = y
    cpl,_,_ = count_per_level([],class_,features,coords,scaler,w, preds, img_nr, boxes,levels[level], '',function)
    if jans_idea:    
        cpl = max(0,cpl)
    loss = cpl - tru
    if len(levels[level]) == 1:
#        print [levels[level]], feat
#        print x[levels[level]]
#        print x[levels[level]][feat]
        return (loss *(-x[levels[level]][0][feat])), []
    count_per_level_temp = 0
    if function != []:
        for fun in function:
            if 'p' in fun[1]:
                if '+' in fun[0]:
                    count_per_level_temp += x[fun[2]][feat]
                elif '-' in fun[0]:
                    count_per_level_temp -= x[fun[2]][feat]
                else:
                    print 'wrong symbol', fun[0]
            elif 'f' in fun[1]:
                if '+' in fun[0]:
                    count_per_level_temp += features[fun[2]][feat]
                elif '-' in fun[0]:
                    count_per_level_temp -= features[fun[2]][feat]
                else:
                    print 'wrong symbol 0', fun[0]
            else:
                print 'wrong symbol 1', fun[1]
        return loss *(- count_per_level_temp), function
    else:
        if feat != 0:
            print 'else, although feat > 0'
        level_boxes = []
        for i in levels[level]:
            level_boxes.append(boxes[i][0])
            
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
                     count_per_level_temp += x[ind][feat]
                     function.append(['+','p',ind])
                 else:
                     count_per_level_temp -= x[ind][feat]
                     function.append(['-','p',ind])
              else:
                 if feat_exist == True:
                    if I in coords and I != []:
                      ind = coords.index(I)
                      if len(base)%2==1:
                         count_per_level_temp += features[ind][feat]
                         function.append(['+','f',ind])
                      else:
                         count_per_level_temp -= features[ind][feat]
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
        return loss * (- count_per_level_temp), function