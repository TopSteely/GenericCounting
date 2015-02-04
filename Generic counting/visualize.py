import os
from matplotlib.pylab import colorbar
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pylab as pl
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import matplotlib as mpl


def bool_rect_intersect(A, B):
    return not (B[0]>A[2] or B[2]<A[0] or B[3]<A[1] or B[1]>A[3])     
        #return !(r2.left > r1.right || r2.right < r1.left || r2.top > r1.bottom ||r2.bottom < r1.top);


def overlap(A, B):
    left = max(A[0], B[0]);
    right = min(A[2], B[2]);
    bottom = max(A[1], B[1]);
    top = min(A[3], B[3]);
    intersection = [left, bottom, right, top];
    surface_intersection = (intersection[2]-intersection[0])*(intersection[3]-intersection[1]);
    surface_ground_truth = (A[2]- A[0])*(A[3]-A[1]);
    # round to 0.1 decimal
    p = surface_intersection / surface_ground_truth;
    if not bool_rect_intersect(A, B):
        p = 0
    if p<0:
        p=0
    return intersection, p
    
    
class_ = 'sheep'
c = 'partial'
a = 0.001
ets = 0.00001
baseline = False
# get labels, features
test_imgs, train_imgs = get_seperation()

if os.path.isfile('/home/t/Schreibtisch/Thesis/Models/'+class_+c+str(a)+str(ets)+'_baseline.pickle') and baseline == True:
    with open('/home/t/Schreibtisch/Thesis/Models/'+class_+c+str(a)+str(ets)+'_baseline.pickle', 'rb') as handle:
        clf = pickle.load(handle)
elif os.path.isfile('/home/t/Schreibtisch/Thesis/Models/'+c+str(a)+str(ets)+'_baseline.pickle') and baseline == True:
    with open('/home/t/Schreibtisch/Thesis/Models/'+c+str(a)+str(ets)+'_baseline.pickle', 'rb') as handle:
        clf = pickle.load(handle)
elif os.path.isfile('/home/t/Schreibtisch/Thesis/Models/'+class_+c+str(a)+str(ets)+'.pickle') and baseline == False:
    with open('/home/t/Schreibtisch/Thesis/Models/'+class_+c+str(a)+str(ets)+'.pickle', 'rb') as handle:
        clf = pickle.load(handle)
elif os.path.isfile('/home/t/Schreibtisch/Thesis/Models/'+c+str(a)+str(ets)+'.pickle') and baseline == False:
    with open('/home/t/Schreibtisch/Thesis/Models/'+c+str(a)+str(ets)+'.pickle', 'rb') as handle:
        clf = pickle.load(handle)
for minibatch in range(0,200,1):
    X_p_test, y_p_test, investigate = get_data(class_, test_imgs, train_imgs, minibatch, minibatch + 1, 'test', c)
    pred = clf.predict(X_p_test)
    cNorm  = colors.Normalize(vmin=min(pred), vmax=max(pred))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=pl.cm.jet)
    scalarMap.set_array(range(int(min(pred)), int(max(pred))))
    investigate.append(pred)
    img = investigate[0][0]
    if os.path.isfile('/home/t/Schreibtisch/Thesis/SS_Boxes/'+ (format(img, "06d")) +'.txt'):
        f = open('/home/t/Schreibtisch/Thesis/SS_Boxes/'+ (format(img, "06d")) +'.txt', 'r')
    if os.path.isfile('/home/t/Schreibtisch/Thesis/Rois/GroundTruth/'+ (format(img, "06d")) +'.txt'):
        gr = open('/home/t/Schreibtisch/Thesis/Rois/GroundTruth/'+ (format(img, "06d")) +'.txt', 'r')
    ground_truths = []
    for line in gr:
        tmp = line.split(',')
        ground_truth = []
        for s in tmp:
            # -1 because of matlab -> python
            ground_truth.append(int(s) - 1)
        ground_truths.append(ground_truth)
    objects = ground_truths.__len__()
    img = mpimg.imread('/home/t/Schreibtisch/Thesis/VOCdevkit1/VOC2007/JPEGImages/'+ (format(img, "06d")) +'.jpg')
#    if objects == 1:
#         _, ax = plt.subplots(2, 2)
#         ax[0, 0].imshow(img)
#    else:
#        if objects < 4:
#            _, ax = plt.subplots(2, 3)
#            rows = 2
#        if objects > 3 and objects < 7:
#            _, ax = plt.subplots(3, 3)
#            rows = 3
    _, ax = plt.subplots(objects + 1, 3)
    ax[0, 0].imshow(img)
    for row, prediction, line in zip(investigate, pred.tolist(), f):
        tmp = line.split(',')
        coord = []
        for s in tmp:
            coord.append(float(s))
        print row, coord, prediction, (int(coord[0]), int(coord[1])), int(coord[2] - coord[0]), int(coord[3] - coord[1])
        # in order to test overlap function
        #ax3.add_patch(Rectangle((int(coord[0]), int(coord[1])), int(coord[2] - coord[0]), int(coord[3] - coord[1]), alpha=0.4, facecolor = 'yellow', edgecolor = 'yellow'))   
        #ax3.add_patch(Rectangle((int(ground_truths[0][0]), int(ground_truths[0][1])), int(ground_truths[0][2] - ground_truths[0][0]), int(ground_truths[0][3] - ground_truths[0][1]), alpha=0.4, facecolor = 'red', edgecolor = 'red'))   
        #ax3.add_patch(Rectangle((int(intersection[0]), int(intersection[1])), int(intersection[2] - intersection[0]), int(intersection[3] - intersection[1]), alpha=0.4, facecolor = 'blue', edgecolor = 'blue'))   
        #print p
        colorVal = scalarMap.to_rgba(prediction)
        overlapped = False
        for i in range(objects):
            _, p = overlap(ground_truths[i], coord)
            if p > 0:
                if objects == 1:
                    ax[1, 0].add_patch(Rectangle((int(coord[0]), int(coord[1])), int(coord[2] - coord[0]), int(coord[3] - coord[1]), alpha=0.1, facecolor = colorVal, edgecolor = colorVal))
                else:
                    if i < 3:
                        ax[1, i].add_patch(Rectangle((int(coord[0]), int(coord[1])), int(coord[2] - coord[0]), int(coord[3] - coord[1]), alpha=0.1, facecolor = colorVal, edgecolor = colorVal))
                    if i >= 3:
                        ax[2, i%3].add_patch(Rectangle((int(coord[0]), int(coord[1])), int(coord[2] - coord[0]), int(coord[3] - coord[1]), alpha=0.1, facecolor = colorVal, edgecolor = colorVal))
                overlapped = True
            if p >= 0.9:
                if objects == 1:
                    ax[1, 1].add_patch(Rectangle((int(coord[0]), int(coord[1])), int(coord[2] - coord[0]), int(coord[3] - coord[1]), alpha=0.1, facecolor = colorVal, edgecolor = colorVal))
                else:
                    ax[0, 2].add_patch(Rectangle((int(coord[0]), int(coord[1])), int(coord[2] - coord[0]), int(coord[3] - coord[1]), alpha=0.1, facecolor = colorVal, edgecolor = colorVal))
                overlapped = True
        if overlapped == False:
            if objects == 1:
                ax[0, 1].add_patch(Rectangle((int(coord[0]), int(coord[1])), int(coord[2] - coord[0]), int(coord[3] - coord[1]), alpha=0.1, facecolor = colorVal, edgecolor = colorVal))   
            else:
                ax[0, 1].add_patch(Rectangle((int(coord[0]), int(coord[1])), int(coord[2] - coord[0]), int(coord[3] - coord[1]), alpha=0.1, facecolor = colorVal, edgecolor = colorVal))   
    if objects == 1:
        ax[0, 0].get_xaxis().set_visible(False)
        ax[0, 0].get_yaxis().set_visible(False)
        ax[0, 1].get_xaxis().set_visible(False)
        ax[0, 1].get_yaxis().set_visible(False)
        ax[1, 0].get_xaxis().set_visible(False)
        ax[1, 0].get_yaxis().set_visible(False)
        ax[1, 1].get_xaxis().set_visible(False)
        ax[1, 1].get_yaxis().set_visible(False)
        ax[0, 1].add_patch(Rectangle((int(ground_truths[0][0]), int(ground_truths[0][1])), int(ground_truths[0][2] - ground_truths[0][0]), int(ground_truths[0][3] - ground_truths[0][1]), alpha=0.8, facecolor = 'None', edgecolor = 'black'))                
        ax[1, 0].add_patch(Rectangle((int(ground_truths[0][0]), int(ground_truths[0][1])), int(ground_truths[0][2] - ground_truths[0][0]), int(ground_truths[0][3] - ground_truths[0][1]), alpha=0.8, facecolor = 'None', edgecolor = 'black'))                
        ax[1, 1].add_patch(Rectangle((int(ground_truths[0][0]), int(ground_truths[0][1])), int(ground_truths[0][2] - ground_truths[0][0]), int(ground_truths[0][3] - ground_truths[0][1]), alpha=0.8, facecolor = 'None', edgecolor = 'black'))                
        ax[0, 1].set_title('Non-overlapping boxes')
        ax[1, 1].set_title('Boxes with intersection of at least 0.9')
        ax[1, 0].set_title('Boxes overlapping with object #' + str(i - 1))
        ax[0, 1].set_ylim(ax[0, 0].get_ylim())
        ax[0, 1].set_xlim(ax[0, 0].get_xlim())
        ax[0, 1].set_yscale(ax[0, 0].get_yscale())
        ax[0, 1].set_xscale(ax[0, 0].get_xscale())
        ax[0, 1].set_aspect(ax[0, 0].get_aspect())
        ax[1, 1].set_ylim(ax[0, 0].get_ylim())
        ax[1, 1].set_xlim(ax[0, 0].get_xlim())
        ax[1, 1].set_yscale(ax[0, 0].get_yscale())
        ax[1, 1].set_xscale(ax[0, 0].get_xscale())
        ax[1, 1].set_aspect(ax[0, 0].get_aspect())
        ax[1, 0].set_ylim(ax[0, 0].get_ylim())
        ax[1, 0].set_xlim(ax[0, 0].get_xlim())
        ax[1, 0].set_yscale(ax[0, 0].get_yscale())
        ax[1, 0].set_xscale(ax[0, 0].get_xscale())
        ax[1, 0].set_aspect(ax[0, 0].get_aspect())
    else:
        for o in range(objects):
            ax[0, 1].add_patch(Rectangle((int(ground_truths[o-1][0]), int(ground_truths[o-1][1])), int(ground_truths[o-1][2] - ground_truths[o-1][0]), int(ground_truths[o-1][3] - ground_truths[o-1][1]), alpha=0.8, facecolor = 'None', edgecolor = 'black'))
            ax[0, 2].add_patch(Rectangle((int(ground_truths[o-1][0]), int(ground_truths[o-1][1])), int(ground_truths[o-1][2] - ground_truths[o-1][0]), int(ground_truths[o-1][3] - ground_truths[o-1][1]), alpha=0.8, facecolor = 'None', edgecolor = 'black'))                 
        ax[0, 1].set_title('Non-overlapping boxes')
        ax[0, 2].set_title('Boxes with intersection of at least 0.9')
        for i in range(objects):
            if i < 3:
                ax[1, i].add_patch(Rectangle((int(ground_truths[i][0]), int(ground_truths[i][1])), int(ground_truths[i][2] - ground_truths[i][0]), int(ground_truths[i][3] - ground_truths[i][1]), alpha=0.8, facecolor = 'None', edgecolor = 'black'))
                ax[1, i].set_title('Boxes overlapping with object #' + str(i))
            if i >= 3:
                ax[2, i%3].add_patch(Rectangle((int(ground_truths[i][0]), int(ground_truths[i][1])), int(ground_truths[i][2] - ground_truths[i][0]), int(ground_truths[i][3] - ground_truths[i][1]), alpha=0.8, facecolor = 'None', edgecolor = 'black'))
                ax[2, i%3].set_title('Boxes overlapping with object #' + str(i))
        for i in range(rows):
            for j in range(3):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)
                ax[i, j].set_ylim(ax[0, 0].get_ylim())
                ax[i, j].set_xlim(ax[0, 0].get_xlim())
                ax[i, j].set_yscale(ax[0, 0].get_yscale())
                ax[i, j].set_xscale(ax[0, 0].get_xscale())
                ax[i, j].set_aspect(ax[0, 0].get_aspect())
        #ax2.add_patch(Rectangle((int(coord[0]), int(coord[1])), int(coord[2] - coord[0]), int(coord[3] - coord[1]), alpha=0.1, facecolor = colorVal, edgecolor = colorVal))   
    colorbar(scalarMap, shrink=0.9)
    plt.draw()
    #break


