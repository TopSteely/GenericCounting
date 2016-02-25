from itertools import chain, islice
import networkx as nx
from get_intersection import get_intersection
import itertools
from get_intersection_count import get_intersection_count
from collections import deque
from get_intersection import get_intersection

def get_set_intersection(set_):
    intersection = get_intersection(set_[0], set_[1])
    if intersection == []:
       return []
    for s in set_[2:len(set_)]:
        intersection = get_intersection(intersection, s)
        if intersection == []:
            return []
    return intersection


def train_per_level(features,coords,scaler, x,node,img_nr, boxes, children, parent, feat, function):
    if len(children) == 1:
        return (-x[node][feat] + x[parent][feat])
    count_per_level = 0
    if function != []:
        for fun in function:
            if fun[1] == 0:
                if fun[0] == 0:
                    count_per_level += x[fun[2]][feat]
                elif fun[0] == 1:
                    count_per_level -= x[fun[2]][feat]
                else:
                    print 'wrong symbol', fun[0]
            if fun[1] == 1:
                if fun[0] == 0:
                    count_per_level += features[fun[2]][feat]
                elif fun[0] == 1:
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
                     function.append([0,0,ind])
                 else:
                     count_per_level -= x[ind][feat]
                     function.append([1,0,ind])
              else:
                 if feat_exist == True:
                    if I in coords and I != []:
                      ind = coords.index(I)
                      if len(base)%2==1:
                         count_per_level += features[ind][feat]
                         function.append([0,1,ind])
                      else:
                         count_per_level -= features[ind][feat]
                         function.append([1,1,ind])
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