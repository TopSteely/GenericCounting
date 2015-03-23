import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import itertools


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
boxes = boxes[0:59]
intersection_counts = np.zeros((len(boxes), len(boxes)))
pairwise_boxes = np.zeros((len(boxes), len(boxes)))
fully_overlapped = np.zeros((len(boxes), len(boxes)))
p_boxes = np.zeros((len(boxes)))
fig = plt.figure()
plt.imshow(img)
cur = plt.gca()
sums = np.zeros(len(boxes))
sums_ = 0
no_intersection = False
no_intersections = []



it = 2
print 'boxes:', len(boxes)
while no_intersection == False:
    combinations = list(itertools.combinations(range(len(boxes)), it))
    len_ = len(combinations)
    print it, len_
#    # TODO: takes too long.....    
#    if it > 2:
#        print 'start'
#        combinations[:] = [x for x in combinations if intersectable(x, no_intersections)]
#        print 'end'
#    len_ = len(combinations)

    for comb, comb_i in zip(combinations, range(len_)):
        if it > 3:
            if comb_i % int(len_/10.0)  == 0:
                print comb_i, len_
        set_ = []
        for c in comb:
            set_.append(boxes[c][0])
        I = get_set_intersection(set_)
        if I != []:
            sums[it-2] += get_intersection_count(I, ground_truths)
        if it == 2:
            if I == []:
                no_intersections.append(comb)
    if sums[it-2] == 0:
        no_intersection == True
        break
    it += 1

for box,i in zip(boxes, range(len(boxes))):
    cur.add_patch(Rectangle((int(box[0][0]), int(box[0][1])), int(box[0][2] - box[0][0]), int(box[0][3] - box[0][1]), alpha = 0.3, facecolor = 'red', edgecolor = 'red'))
    p_boxes[i] = box[1]
# http://mathworld.wolfram.com/Inclusion-ExclusionPrinciple.html
# union of all boxes: sum of all boxes - sum pairwise_intersections + sum 3 set intersections - sum 4 set intersections
objects =  iep(p_boxes, sums)
plt.show()
print sums
fig.suptitle('Baseline 4 - objects: %0.1f, prediction: %0.2f' % (boxes[0][1], objects))