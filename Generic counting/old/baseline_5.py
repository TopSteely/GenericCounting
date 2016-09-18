import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import itertools
import cProfile
#import pyximport; pyximport.install(pyimport = True)
#import qsort
#import f5
import networkx as nx
from collections import deque
from itertools import chain, islice
import time


def enumerate_all_cliques(G):
    """Returns all cliques in an undirected graph.

    This method returns cliques of size (cardinality)
    k = 1, 2, 3, ..., maxDegree - 1.

    Where maxDegree is the maximal degree of any node in the graph.

    Parameters
    ----------
    G: undirected graph

    Returns
    -------
    generator of lists: generator of list for each clique.

    Notes
    -----
    To obtain a list of all cliques, use
    :samp:`list(enumerate_all_cliques(G))`.

    Based on the algorithm published by Zhang et al. (2005) [1]_
    and adapted to output all cliques discovered.

    This algorithm is not applicable on directed graphs.

    This algorithm ignores self-loops and parallel edges as
    clique is not conventionally defined with such edges.

    There are often many cliques in graphs.
    This algorithm however, hopefully, does not run out of memory
    since it only keeps candidate sublists in memory and
    continuously removes exhausted sublists.

    References
    ----------
    .. [1] Yun Zhang, Abu-Khzam, F.N., Baldwin, N.E., Chesler, E.J.,
           Langston, M.A., Samatova, N.F.,
           Genome-Scale Computational Approaches to Memory-Intensive
           Applications in Systems Biology.
           Supercomputing, 2005. Proceedings of the ACM/IEEE SC 2005
           Conference, pp. 12, 12-18 Nov. 2005.
           doi: 10.1109/SC.2005.29.
           http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1559964&isnumber=33129
    """
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
        yield base
        for i, u in enumerate(cnbrs):
            # Use generators to reduce memory consumption.
            queue.append((chain(base, [u]),
                          filter(nbrs[u].__contains__,
                                 islice(cnbrs, i + 1, None))))
                                 
def sum_overlapped(fully_overlapped, intersection_counts):
    assert len(fully_overlapped) == len(intersection_counts)
    ret = 0.0
    for i in range(len(intersection_counts)):
        for j in range(len(intersection_counts[0])):
            ret += intersection_counts[i,j] * fully_overlapped[i,j]
    return ret


def bool_rect_intersect(A, B):
    return not (B[0]>A[2] or B[2]<A[0] or B[3]<A[1] or B[1]>A[3])
        #return !(r2.left > r1.right || r2.right < r1.left || r2.top > r1.bottom ||r2.bottom < r1.top);
    
    
def get_overlap_ratio(A, B):
    in_ = bool_rect_intersect(A, B)
    if not in_:
        return 0
    else:
        left = max(A[0], B[0]);
        top = max(A[1], B[1]);
        right = min(A[2], B[2]);
        bottom = min(A[3], B[3]);
        intersection = [left, top, right, bottom];
        surface_intersection = (intersection[2]-intersection[0])*(intersection[3]-intersection[1]);
        surface_A = (A[2]- A[0])*(A[3]-A[1]) + 0.0;
        return surface_intersection / surface_A
       
       
def get_intersection(A, B):
    in_ = bool_rect_intersect(A, B)
    if not in_:
        return []
    else:
        left = max(A[0], B[0]);
        right = min(A[2], B[2]);
        top = max(A[1], B[1]);
        bottom = min(A[3], B[3]);
        intersection = [left, top, right, bottom];
        return intersection
    

def get_intersection_count(I, gts):
    l = 0.0
    for gr in gts:
        p = get_overlap_ratio(gr, I)
        l += p
    return l
    

def get_set_intersection(set_):
    intersection = get_intersection(set_[0], set_[1])
    if intersection == []:
       return []
    for s in set_[2:len(set_)]:
        intersection = get_intersection(intersection, s)
        if intersection == []:
            return []
    return intersection
    
    
def iep(p_boxes, sums):
    ret = sum(p_boxes)
    for summe, ij in zip(sums, range(len(sums))):
        if ij % 2 == 0:
            ret -= summe
        else:
            ret += summe
    return ret
    
    
def intersectable(x, no_intersections):
    for iter_ in itertools.combinations(x, 2):
        if iter_ in no_intersections:
            return False
    return True 
    
    
def prof():  
    img = 475
    c = 'partial'
    class_ = 'sheep'
    boxes = []
    coords = []
    test_imgs, train_imgs = get_seperation()
    X_p_test, y_p_test, investigate = get_data(class_, test_imgs, train_imgs, 6, 7, 'test', c)
    if os.path.isfile('/home/t/Schreibtisch/Thesis/SS_Boxes/'+ (format(img, "06d")) +'.txt'):
        f = open('/home/t/Schreibtisch/Thesis/SS_Boxes/'+ (format(img, "06d")) +'.txt', 'r')
    if os.path.isfile('/home/t/Schreibtisch/Thesis/Rois/GroundTruth/'+ class_ + '_' + (format(img, "06d")) +'.txt'):
        gr = open('/home/t/Schreibtisch/Thesis/Rois/GroundTruth/'+ class_ + '_' + (format(img, "06d")) +'.txt', 'r')
    ground_truths = []
    for line in gr:
        tmp = line.split(',')
        ground_truth = []
        for s in tmp:
            # -1 because of matlab -> python
            ground_truth.append(int(s) - 1)
        ground_truths.append(ground_truth)
    img = mpimg.imread('/home/t/Schreibtisch/Thesis/VOCdevkit1/VOC2007/JPEGImages/'+ (format(img, "06d")) +'.jpg')
    for line, y in zip(f, investigate):
        tmp = line.split(',')
        coord = []
        for s in tmp:
            coord.append(float(s))
        boxes.append([coord, y[2]])
        coords.append(coord)
    boxes = boxes[0:45]
    p_boxes = np.zeros((len(boxes)))
    fig = plt.figure()
    plt.imshow(img)
    cur = plt.gca()
    sums = np.zeros(len(boxes))
    
    
    
    print 'boxes:', len(boxes)
    combinations = list(itertools.combinations(range(len(boxes)), 2)) 
    G = nx.Graph()
    G.add_edges_from(combinations)
    for comb in combinations:
        set_ = []
        for c in comb:
            set_.append(boxes[c][0])
        I = get_set_intersection(set_)
        if I == []:
            G.remove_edges_from([comb])
    print 'edges: ', G.number_of_edges()
    cliques = enumerate_all_cliques(G)
    gen = (clq for clq in cliques if len(clq)>1)
    count_cliques = 0
    for clq in gen:
        I = [0,0,1000,1000]
        for c in clq:
            I = get_intersection(boxes[c][0], I)
        if I != []: #always I, since clique, right?
            sums[len(clq)-2] += get_intersection_count(I, ground_truths)
        count_cliques += 1
    print 'Cliques:', count_cliques
    for box,i in zip(boxes, range(len(boxes))):
        cur.add_patch(Rectangle((int(box[0][0]), int(box[0][1])), int(box[0][2] - box[0][0]), int(box[0][3] - box[0][1]), alpha = 0.3, facecolor = 'red', edgecolor = 'red'))
        p_boxes[i] = box[1]
    # http://mathworld.wolfram.com/Inclusion-ExclusionPrinciple.html
    # union of all boxes: sum of all boxes - sum pairwise_intersections + sum 3 set intersections - sum 4 set intersections
    objects =  iep(p_boxes, sums)
    plt.show()
    fig.suptitle('Baseline 4 - objects: %0.1f, prediction: %0.2f' % (boxes[0][1], objects))
    print sums
    print objects
    plt.figure()
    nx.draw(G)
    plt.draw()

start_time = time.time()
prof()
tzs = time.time() - start_time    
print("--- %s seconds ---" % (tzs))
#prof()