import os
import sys
import gzip
import pickle
import numpy as np
from scipy.misc import imread, imresize
from load import get_seperation, get_traineval_seperation, get_labels, get_features

import theano
from theano import tensor as T

import lasagne
#from lasagne.updates import nesterov_momentum, adam
from lasagne.layers import helper

from sklearn.preprocessing import MinMaxScaler

def learn_root(w,x,y,learning_rate,alphas):
    inner_prod = 0.0
    for f_ in range(len(w)):
        inner_prod += (w[f_] * x[f_])
    # clip negative predictions
    #dloss = inner_prod - y
    dloss = max(0,inner_prod) - y
    for f_ in range(len(w)):
        w[f_] += (learning_rate * ((-x[f_] * dloss)) + alphas[1] * w[f_])
    return w

variant = 'bottleneck'
depth = 18
width = 1
print 'Using %s ResNet with depth %d and width %d.'%(variant,depth,width)

if variant == 'normal':
    from models.models import ResNet_FullPreActivation as ResNet
elif variant == 'bottleneck':
    from models.models import ResNet_BottleNeck_FullPreActivation as ResNet
elif variant == 'wide':
    from models.models import ResNet_FullPre_Wide as ResNet
else:
    print ('Unsupported model %s' % variant)

BATCHSIZE = 1

'''
Set up all theano functions
'''
X = T.tensor4('X')
Y = T.ivector('y')

# set up theano functions to generate output by feeding data through network, any test outputs should be deterministic
# load model
if width > 1:
    output_layer = ResNet(X, n=depth, k=width)
else:
    output_layer = ResNet(X, n=depth)
output_test = lasagne.layers.get_output(output_layer, deterministic=True)

# set up training and prediction functions
features = theano.function(inputs=[X], outputs=output_test)

# load network weights
f = gzip.open('models/data/weights/resnet164_fullpreactivation.pklz', 'r')
all_params = pickle.load(f)
f.close()
helper.set_all_param_values(output_layer, all_params)
test_imgs, train_imgs = get_seperation()
#train_imgs = train_imgs[0:37]
#test_imgs = test_imgs[0:37]
img_train, img_eval = get_traineval_seperation(train_imgs)
class_ = 'sheep'
w1 = 0.01 * np.random.rand(10)
w2 = 0.01 * np.random.rand(4096)
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

#normalize
for img_nr in img_train:
    #load image
    if os.path.isfile('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg'):
        img = imread('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg')
    else:
        print 'warning: /var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg doesnt exist'
    img = imresize(img,[32,32])
    train_X = np.zeros((1,3,32,32))
    train_X[:,0,:,:] = img[:,:,0]
    train_X[:,1,:,:] = img[:,:,1]
    train_X[:,2,:,:] = img[:,:,2]
    #get features
    feat_1 = features(np.array(train_X,dtype=np.float32))
    feat_2 = get_features(img_nr, 1)

    scaler1.partial_fit(feat_1)  
    scaler2.partial_fit(feat_2)  

    

#train
for img_nr in img_train:
    #load image
    if os.path.isfile('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg'):
        img = imread('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg')
    else:
        print 'warning: /var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg doesnt exist'
    img = imresize(img,[32,32])
    train_X = np.zeros((1,3,32,32))
    train_X[:,0,:,:] = img[:,:,0]
    train_X[:,1,:,:] = img[:,:,1]
    train_X[:,2,:,:] = img[:,:,2]
    #get features
    feat_1 = features(np.array(train_X,dtype=np.float32))
    feat_1 = scaler1.transform(feat_1)
    
    feat_2 = get_features(img_nr, 1)
    feat_2 = scaler2.transform(feat_2)
    
    #get label
    y_p = get_labels(class_,img_nr, 'partial', 1)
    #train
    w1 = learn_root(w1,feat_1[0],y_p[0],0.001,[0,0.00001])
    w2 = learn_root(w2,feat_2[0],y_p[0],0.001,[0,0.00001])
    
#compute loss        
mse_1 = []
mse_2 = []
for img_nr in test_imgs:
    print img_nr
    #load image
    if os.path.isfile('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg'):
        img = imread('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg')
    else:
        print 'warning: /var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg doesnt exist'
    img = imresize(img,[32,32])
    train_X = np.zeros((1,3,32,32))
    train_X[:,0,:,:] = img[:,:,0]
    train_X[:,1,:,:] = img[:,:,1]
    train_X[:,2,:,:] = img[:,:,2]
    #get features
    feat_1 = features(np.array(train_X,dtype=np.float32))
    feat_1 = scaler1.transform(feat_1)
    
    feat_2 = get_features(img_nr, 1)
    feat_2 = scaler2.transform(feat_2)
    #get label
    y_p = get_labels(class_,img_nr, 'partial', 1)
    mse_1.append((max(0,np.dot(w1, feat_1[0])) - y_p[0])**2)
    mse_2.append((max(0,np.dot(w2, feat_2[0])) - y_p[0])**2)
    print y_p[0], mse_1[-1],mse_2[-1]
    
print "new Theano features: ", np.array(mse_1).sum() / len(mse_1)
print "old Dennis features: ", np.array(mse_2).sum() / len(mse_2)
