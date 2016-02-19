from sklearn import linear_model
import matplotlib
matplotlib.use('agg')
import math
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import itertools
import pickle
import networkx as nx
import pyximport; pyximport.install(pyimport = True)
import get_overlap_ratio
from get_intersection import get_intersection
from collections import deque
from itertools import chain, islice
from get_intersection_count import get_intersection_count

def count_per_level(scaler,w,preds,img, boxes, boxes_level, mode):
    #tested
    sums = np.zeros(len(boxes_level))
    if len(boxes_level) == 1:
        return preds[boxes_level[0]],None
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
    sums,used_boxes = sums_of_all_cliques(scaler,w, G, preds, boxes, sums, img, mode)
    return iep(sums),used_boxes
    
    
def iep(sums):
    ret = 0
    for summe, ij in zip(sums, range(len(sums))):
        if ij % 2 == 0:
            ret += summe
        else:
            ret -= summe
    return ret
    
    
def sums_of_all_cliques(scaler,w, G, preds, boxes, sums, img_nr, mode):
    feat_exist = True #must be false in order to write
    real_b = [b[0] for b in boxes]
    used_boxes = []
    write_coords = []
    length = 1
    index = {}
    nbrs = {}
    coords = []
    features = []
    if mode != 'gt':
        if os.path.isfile('/home/t/Schreibtisch/Thesis/sheep'+ (format(img_nr, "06d")) +'_coords.txt'):
            f_c = open('/home/t/Schreibtisch/Thesis/sheep'+ (format(img_nr, "06d")) +'_coords.txt', 'r+')
            for i,line in enumerate(f_c):
                str_ = line.rstrip('\n').split(',')
                cc = []
                for s in str_:
                   cc.append(float(s))
                coords.append(cc)
        else:
            print '/home/t/Schreibtisch/Thesis/sheep'+ (format(img_nr, "06d")) +'_coords.txt'
            f_c = open('/home/stahl/Features_prop_windows/upper_levels/sheep'+ (format(img_nr, "06d")) +'.txt', 'w+')
    
        if os.path.isfile('/home/t/Schreibtisch/Thesis/sheep'+ (format(img_nr, "06d")) +'_feat.txt'):
            f = open('/home/t/Schreibtisch/Thesis/sheep'+ (format(img_nr, "06d")) +'_feat.txt', 'r') 
            feat_exist = True
            for i,line in enumerate(f):
                str_ = line.rstrip('\n').split(',')  
                ff = []
                for s in str_:
                   ff.append(float(s))
                features.append(ff)
            if less_features:
                features = [fts[0:features_used] for fts in features]
            features = scaler.transform(features)
        else:
            print '/home/stahl/Features_prop_windows/Features_upper/sheep'+ (format(img_nr, "06d")) +'.txt doesnt exist'
            
        assert len(coords) == len(features)
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
             used_boxes.append([I,np.dot(w,features[ind])])
          else:
             if feat_exist == True:
                if mode != 'gt':
                    if I in coords and I != []:
                      ind = coords.index(I)
                      sums[len(base)-1] += np.dot(w,features[ind])
                      used_boxes.append([I,np.dot(w,features[ind])])
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

    if feat_exist == False:
       print 'write coords', len(write_coords)
       for coor in write_coords:
          f_c.seek(0,2)
          f_c.write(str(coor[0])+','+str(coor[1])+','+str(coor[2])+','+str(coor[3]))
          f_c.write('\n')
    return sums, used_boxes         

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
                                 
def sum_overlapped(fully_overlapped, intersection_counts):
    assert len(fully_overlapped) == len(intersection_counts)
    ret = 0.0
    for i in range(len(intersection_counts)):
        for j in range(len(intersection_counts[0])):
            ret += intersection_counts[i,j] * fully_overlapped[i,j]
    return ret
    

def get_set_intersection(set_):
    intersection = get_intersection(set_[0], set_[1])
    if intersection == []:
       return []
    for s in set_[2:len(set_)]:
        intersection = get_intersection(intersection, s)
        if intersection == []:
            return []
    return intersection
    
    
def iep(sums):
    ret = 0
    for summe, ij in zip(sums, range(len(sums))):
        if ij % 2 == 0:
            ret += summe
        else:
            ret -= summe
    return ret
    
    
def intersectable(x, no_intersections):
    for iter_ in itertools.combinations(x, 2):
        if iter_ in no_intersections:
            return False
    return True
    
    
def sort_boxes(boxes, preds, from_, to):
    sorted_boxes = []
    sorted_preds = []
    decorated = [((box[0][3]-box[0][1])*(box[0][2]-box[0][0]), i) for i, box in enumerate(boxes)]
    decorated.sort()
    for box, i in reversed(decorated):
        sorted_boxes.append(boxes[i])
        sorted_preds.append(preds[i])
    return sorted_boxes[from_:to], sorted_preds[from_:to]
    
    
def surface_area(boxes, boxes_level):
    if len(boxes_level) == 1:
        I = boxes[boxes_level[0]][0]
        return (I[3]-I[1])*(I[2]-I[0])
    surface_area = 0
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
                surface_area += (I[3]-I[1])*(I[2]-I[0])
        elif len(base)%2==0:
            surface_area -= (I[3]-I[1])*(I[2]-I[0])
                
        for i, u in enumerate(cnbrs):
            # Use generators to reduce memory consumption.
            queue.append((chain(base, [u]),
                          filter(nbrs[u].__contains__,
                                 islice(cnbrs, i + 1, None)))) 
            
    return surface_area
    
    
def count_per_level_(boxes, ground_truths, boxes_level):
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
    
    
    
#@profile   
def prof():  
    img = 107
    c = 'partial'
    class_ = 'sheep'
    boxes = []
    coords = []
    alphas = [1,0,1,0]
    learning_rate0 = math.pow(10,-3)
    test_imgs, train_imgs = get_seperation()
    X_p_test, y_p_test, investigate = get_data(class_, test_imgs, train_imgs, 0, 1, 'training', c)
    
    if os.path.isfile('/home/t/Schreibtisch/Thesis/Models/'+class_+c+'%s_%s_%s_%s_%s.pickle'%(learning_rate0, alphas[0],alphas[1],alphas[2],alphas[3])):
        with open('/home/t/Schreibtisch/Thesis/Models/'+class_+c+'%s_%s_%s_%s_%s.pickle'%(learning_rate0, alphas[0],alphas[1],alphas[2],alphas[3]), 'rb') as handle:
            w = pickle.load( handle)
    if os.path.isfile('/home/t/Schreibtisch/Thesis/Models/'+class_+c+'%s_%s_%s_%s_%s_scaler.pickle'%(learning_rate0, alphas[0],alphas[1],alphas[2],alphas[3])):
        with open('/home/t/Schreibtisch/Thesis/Models/'+class_+c+'%s_%s_%s_%s_%s_scaler.pickle'%(learning_rate0, alphas[0],alphas[1],alphas[2],alphas[3]), 'rb') as handle:
            scaler = pickle.load( handle)
    X_p_test = scaler.transform(X_p_test)
    preds = np.dot(w,np.array(X_p_test).T)
    
    
    img_nr = investigate[0][0]
    if os.path.isfile('/home/t/Schreibtisch/Thesis/SS_Boxes/'+ (format(img_nr, "06d")) +'.txt'):
        f = open('/home/t/Schreibtisch/Thesis/SS_Boxes/'+ (format(img_nr, "06d")) +'.txt', 'r')
    img = mpimg.imread('/home/t/Schreibtisch/Thesis/VOCdevkit1/VOC2007/JPEGImages/'+ (format(img, "06d")) +'.jpg')
    for line, y in zip(f, investigate):
        tmp = line.split(',')
        coord = []
        for s in tmp:
            coord.append(float(s))
        boxes.append([coord, y[2]])
        coords.append(coord)
    boxes, preds = sort_boxes(boxes, preds, 0,500)
    
    sums = np.zeros(len(boxes))
    
    # top-down
    G = nx.Graph()
    G.add_node(0)
    for box, i in zip(boxes[1:len(boxes)], range(1,len(boxes))):
        ind__overlap_smallest = -1
        for box_, ii in zip(boxes[0:i], range(0,i)):
            if get_overlap_ratio.get_overlap_ratio(box[0], box_[0]) == 1:
                #print i, '-', ii
                ind__overlap_smallest = ii
        G.add_edge(i,ind__overlap_smallest)
        
    # bottom-up
    E = nx.Graph()
    for box, i in zip(list(reversed(boxes)), range(len(boxes)-1, -1, -1)):
        for box_, ii in zip(boxes, range(len(boxes)-1)):
            if get_overlap_ratio.get_overlap_ratio(box[0], box_[0]) == 1:
                if i != ii:
                    #print i, '-', ii
                    ind__overlap_smallest = ii
        if ind__overlap_smallest != -1:
            E.add_edge(i,ind__overlap_smallest)
            
            
            
            
     #new approach
    G = nx.Graph()
    levels = {}
    levels[0] = [0]
    G.add_node(0)
    for box, i in zip(boxes[1:len(boxes)], range(1,len(boxes))):
        if (box[0][2]-box[0][0]) * (box[0][3]-box[0][1]) == 0:
            print box
            print 'surface area of box == 0'
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
    cpls = []
    for level in levels:
        cpl = count_per_level(scaler,w,preds,img_nr, boxes, levels[level], '')
        cpls.append(cpl[0])
        
        
    # http://mathworld.wolfram.com/Inclusion-ExclusionPrinciple.html
    # union of all boxes: sum of all boxes - sum pairwise_intersections + sum 3 set intersections - sum 4 set intersections
#    plt.figure()
#    pos=nx.graphviz_layout(G,prog='dot')
#    sm = plt.cm.ScalarMappable(cmap=cm.RdPu, norm=plt.Normalize(vmin=min(preds), vmax=max(preds)))
#    sm.set_array(preds)
#    colorsV = [sm.to_rgba(i) for i in preds]
#    nx.draw(G,pos,node_color=colorsV)
#    plt.colorbar(sm,shrink=0.8)
#    plt.show()
    plt.figure()
    pos=nx.graphviz_layout(G,prog='dot')
    sm = plt.cm.ScalarMappable(cmap=cm.RdPu, norm=plt.Normalize(vmin=min(preds), vmax=max(preds)))
    sm.set_array(preds)
    colorsV = [sm.to_rgba(i) for i in preds]
    offsets,bla = nx.draw_networkx(G, cpls,pos, arrows=False, node_color=colorsV)
    for of,cpl in zip(offsets,cpls):
        plt.text(1800,of,str(round(cpl,2)))
#    plt.ylim(min(cpls)-0.3, max(cpls)+0.3)
    plt.colorbar(sm)
    plt.show()
    plt.title('Tree with prediction as colored nodes - Image 107')
    
baseline = False
class_ = 'sheep'
less_features = False
prof()