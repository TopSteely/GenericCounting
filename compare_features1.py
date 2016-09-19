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
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer, ElemwiseSumLayer, ElemwiseMergeLayer

from sklearn.preprocessing import MinMaxScaler

import caffe
net_caffe = caffe.Net('ResNet-50-model.prototxt','ResNet-50-model.caffemodel',caffe.TEST)

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
    
#convert from caffe to Lasagne    
resnet = {}
resnet['input'] = InputLayer((None,3,224,224))
resnet['conv1'] = ConvLayer(resnet['input'], num_filters=64, filter_size=7, pad=3, flip_filters=False)
resnet['pool1'] = PoolLayer(resnet['conv1'], pool_size=3, stride=2, mode='max', ignore_border=False)
resnet['res2a_branch1'] = ConvLayer(resnet['pool1'], num_filters=256, filter_size=1, pad=0, flip_filters=False)
resnet['res2a_branch2a'] = ConvLayer(resnet['pool1'], num_filters=64, filter_size=1, pad=0, flip_filters=False)
resnet['res2a_branch2b'] = ConvLayer(resnet['res2a_branch2a'], num_filters=64, filter_size=3, pad=1, flip_filters=False)
resnet['res2a_branch2c'] = ConvLayer(resnet['res2a_branch2b'], num_filters=256, filter_size=1, pad=0, flip_filters=False)
resnet['res2a'] = ElemwiseSumLayer(resnet['res2a_branch2c','res2a_branch1'])
resnet['res2b_branch2a'] = ConvLayer(resnet['res2a'], num_filters=64, filter_size=1, pad=0, flip_filters=False)
resnet['res2b_branch2b'] = ConvLayer(resnet['res2b_branch2a'], num_filters=64, filter_size=3, pad=1, flip_filters=False)
resnet['res2b_branch2c'] = ConvLayer(resnet['res2b_branch2b'], num_filters=256, filter_size=1, pad=0, flip_filters=False)
resnet['res2b'] = ElemwiseSumLayer(resnet['res2a','res2b_branch2c'])
resnet['res2c_branch2a'] = ConvLayer(resnet['res2b'], num_filters=64, filter_size=1, pad=0, flip_filters=False)
resnet['res2c_branch2b'] = ConvLayer(resnet['res2c_branch2a'], num_filters=64, filter_size=3, pad=1, flip_filters=False)
resnet['res2c_branch2c'] = ConvLayer(resnet['res2c_branch2b'], num_filters=256, filter_size=1, pad=0, flip_filters=False)
resnet['res2c'] = ElemwiseSumLayer(resnet['res2b','res2c_branch2c'])
resnet['res3a_branch1'] = ConvLayer(resnet['res2c'], num_filters=512, filter_size=1, pad=0, flip_filters=False)
resnet['res3a_branch2a'] = ConvLayer(resnet['res2c'], num_filters=128, filter_size=1, pad=0, flip_filters=False)
resnet['res3a_branch2b'] = ConvLayer(resnet['res3a_branch2a'], num_filters=128, filter_size=3, pad=1, flip_filters=False)
resnet['res3a_branch2c'] = ConvLayer(resnet['res3a_branch2b'], num_filters=512, filter_size=1, pad=0, flip_filters=False)
resnet['res3a'] = ElemwiseSumLayer(resnet['res3a_branch1','res3a_branch2c'])
resnet['res3b_branch2a'] = ConvLayer(resnet['res3a'], num_filters=128, filter_size=1, pad=0, flip_filters=False)
resnet['res3b_branch2b'] = ConvLayer(resnet['res3b_branch2a'], num_filters=128, filter_size=3, pad=1, flip_filters=False)
resnet['res3b_branch2c'] = ConvLayer(resnet['res3b_branch2b'], num_filters=512, filter_size=1, pad=0, flip_filters=False)
resnet['res3b'] = ElemwiseSumLayer(resnet['res3a','res3b_branch2c'])
resnet['res3c_branch2a'] = ConvLayer(resnet['res3b'], num_filters=128, filter_size=1, pad=0, flip_filters=False)
resnet['res3c_branch2b'] = ConvLayer(resnet['res3c_branch2a'], num_filters=128, filter_size=3, pad=1, flip_filters=False)
resnet['res3c_branch2c'] = ConvLayer(resnet['res3c_branch2b'], num_filters=512, filter_size=1, pad=0, flip_filters=False)
resnet['res3c'] = ElemwiseSumLayer(resnet['res3b','res3c_branch2c'])
resnet['res3d_branch2a'] = ConvLayer(resnet['res3c'], num_filters=128, filter_size=1, pad=0, flip_filters=False)
resnet['res3d_branch2b'] = ConvLayer(resnet['res3d_branch2a'], num_filters=128, filter_size=3, pad=1, flip_filters=False)
resnet['res3d_branch2c'] = ConvLayer(resnet['res3d_branch2b'], num_filters=512, filter_size=1, pad=0, flip_filters=False)
resnet['res3d'] = ElemwiseSumLayer(resnet['res3c','res3d_branch2c'])
resnet['res4a_branch1'] = ConvLayer(resnet['res3d'], num_filters=1024, filter_size=1, pad=0, flip_filters=False)
resnet['res4a_branch2a'] = ConvLayer(resnet['res3d'], num_filters=256, filter_size=1, pad=0, flip_filters=False)
resnet['res4a_branch2b'] = ConvLayer(resnet['res4a_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=False)
resnet['res4a_branch2c'] = ConvLayer(resnet['res4a_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=False)
resnet['res4a'] = ElemwiseSumLayer(resnet['res4a_branch1','res4a_branch2c'])
resnet['res4b_branch2a'] = ConvLayer(resnet['res4a'], num_filters=256, filter_size=1, pad=0, flip_filters=False)
resnet['res4b_branch2b'] = ConvLayer(resnet['res4b_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=False)
resnet['res4b_branch2c'] = ConvLayer(resnet['res4b_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=False)
resnet['res4b'] = ElemwiseSumLayer(resnet['res4b_branch2c','res4a'])
resnet['res4c_branch2a'] = ConvLayer(resnet['res4b'], num_filters=256, filter_size=1, pad=0, flip_filters=False)
resnet['res4c_branch2b'] = ConvLayer(resnet['res4c_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=False)
resnet['res4c_branch2c'] = ConvLayer(resnet['res4c_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=False)
resnet['res4c'] = ElemwiseSumLayer(resnet['res4c_branch2c','res4b'])
resnet['res4d_branch2a'] = ConvLayer(resnet['res4c'], num_filters=256, filter_size=1, pad=0, flip_filters=False)
resnet['res4d_branch2b'] = ConvLayer(resnet['res4d_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=False)
resnet['res4d_branch2c'] = ConvLayer(resnet['res4d_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=False)
resnet['res4d'] = ElemwiseSumLayer(resnet['res4d_branch2c','res4c'])
resnet['res4e_branch2a'] = ConvLayer(resnet['res4d'], num_filters=256, filter_size=1, pad=0, flip_filters=False)
resnet['res4e_branch2b'] = ConvLayer(resnet['res4e_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=False)
resnet['res4e_branch2c'] = ConvLayer(resnet['res4e_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=False)
resnet['res4e'] = ElemwiseSumLayer(resnet['res4e_branch2c','res4d'])
resnet['res4f_branch2a'] = ConvLayer(resnet['res4e'], num_filters=256, filter_size=1, pad=0, flip_filters=False)
resnet['res4f_branch2b'] = ConvLayer(resnet['res4f_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=False)
resnet['res4f_branch2c'] = ConvLayer(resnet['res4f_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=False)
resnet['res4f'] = ElemwiseSumLayer(resnet['res4f_branch2c','res4e'])
resnet['res5a_branch1'] = ConvLayer(resnet['res4f'], num_filters=2048, filter_size=1, pad=0, flip_filters=False)
resnet['res5a_branch2a'] = ConvLayer(resnet['res4f'], num_filters=512, filter_size=1, pad=0, flip_filters=False)
resnet['res5a_branch2b'] = ConvLayer(resnet['res5a_branch2a'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
resnet['res5a_branch2c'] = ConvLayer(resnet['res5a_branch2b'], num_filters=2048, filter_size=1, pad=0, flip_filters=False)
resnet['res5a'] = ElemwiseSumLayer(resnet['res5a_branch2c','res5a_branch1'])
resnet['res5b_branch2a'] = ConvLayer(resnet['res5a'], num_filters=512, filter_size=1, pad=0, flip_filters=False)
resnet['res5b_branch2b'] = ConvLayer(resnet['res5a_branch2a'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
resnet['res5b_branch2c'] = ConvLayer(resnet['res5a_branch2b'], num_filters=2048, filter_size=1, pad=0, flip_filters=False)
resnet['res5b'] = ElemwiseSumLayer(resnet['res5b_branch2c','res5a'])    
resnet['res5c_branch2a'] = ConvLayer(resnet['res5b'], num_filters=512, filter_size=1, pad=0, flip_filters=False)
resnet['res5c_branch2b'] = ConvLayer(resnet['res5c_branch2a'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
resnet['res5c_branch2c'] = ConvLayer(resnet['res5c_branch2b'], num_filters=2048, filter_size=1, pad=0, flip_filters=False)
resnet['res5c'] = ElemwiseSumLayer(resnet['res5c_branch2c','res5b']) 
resnet['pool5'] = PoolLayer(resnet['res5c'], pool_size=7, stride=1, mode='ave', ignore_border=False)
resnet['fc1000'] = ElemwiseMergeLayer(resnet['pool5'], num_filter=1000, merge_function='mul')

    
layers_caffe = dict(zip(list(net_caffe._layer_names), net_caffe.layers))

#copy parameters
for name, layer in resnet.items():
    try:
        layer.W.set_value(layers_caffe[name].blobs[0].data)
        layer.b.set_value(layers_caffe[name].blobs[1].data)       
    except AttributeError:
        continue

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
output_caffe = lasagne.layers.get_output(resnet['output'])
features_caffe = theano.function(inputs=[X], output=output_caffe)

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
w3 = 0.01 * np.random.rand(1000)
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler3 = MinMaxScaler()

#normalize
for img_nr in img_train:
    #load image
    if os.path.isfile('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg'):
        img = imread('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg')
    else:
        print 'warning: /var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg doesnt exist'
    img_1 = imresize(img,[32,32])
    img_2 = imresize(img,[224,224])
    train_X = np.zeros((1,3,32,32))
    train_X1 = np.zeros((1,3,224,224))
    train_X[:,0,:,:] = img_1[:,:,0]
    train_X1[:,0,:,:] = img_2[:,:,0]
    train_X[:,1,:,:] = img_1[:,:,1]
    train_X1[:,1,:,:] = img_2[:,:,1]
    train_X[:,2,:,:] = img_1[:,:,2]
    train_X1[:,2,:,:] = img_2[:,:,2]
    #get features
    feat_1 = features(np.array(train_X,dtype=np.float32))
    feat_2 = get_features(img_nr, 1)
    feat_3 = features_caffe(np.array(train_X1,dtype=np.float32))
    print len(feat_3), len(feat_3[0])

    scaler1.partial_fit(feat_1)  
    scaler2.partial_fit(feat_2)  
    scaler3.partial_fit(feat_3)

    
learning_rates = [0.01,0.001,0.0001]
#for eta in learning_rates:
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
    w1 = learn_root(w1,feat_1[0],y_p[0],eta,[0,0.00001])
    w2 = learn_root(w2,feat_2[0],y_p[0],eta,[0,0.00001])
    
#TODO: evaluate model
#for img_nr in img_eval:
#    #load image
#    if os.path.isfile('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg'):
#        img = imread('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg')
#    else:
#        print 'warning: /var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg doesnt exist'
#    img = imresize(img,[32,32])
#    train_X = np.zeros((1,3,32,32))
#    train_X[:,0,:,:] = img[:,:,0]
#    train_X[:,1,:,:] = img[:,:,1]
#    train_X[:,2,:,:] = img[:,:,2]
#    #get features
#    feat_1 = features(np.array(train_X,dtype=np.float32))
#    feat_1 = scaler1.transform(feat_1)
#    
#    feat_2 = get_features(img_nr, 1)
#    feat_2 = scaler2.transform(feat_2)
#    
#    #get label
#    y_p = get_labels(class_,img_nr, 'partial', 1)
#    #train
#    w1 = learn_root(w1,feat_1[0],y_p[0],eta,[0,0.00001])
#    w2 = learn_root(w2,feat_2[0],y_p[0],eta,[0,0.00001])
    
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
