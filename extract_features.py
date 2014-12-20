#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, os.path, numpy, caffe
'''
MEAN_FILE = 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
MODEL_FILE = 'examples/imagenet/imagenet_feature.prototxt'
PRETRAINED = 'examples/imagenet/caffe_reference_imagenet_model'
'''
if len(sys.argv) != 5:
  print "Usage: python extract_features.py MEAN_FILE MODEL_FILE PRETRAINED IMAGE_FILE"
  sys.exit()

caffe_root = '/home/ispick/Projects/caffe/'
MEAN_FILE = sys.argv[1]
#MEAN_FILE = 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
#print '======DEBUG====='
#print MEAN_FILE
MODEL_FILE = sys.argv[2]
PRETRAINED = sys.argv[3]
IMAGE_FILE = sys.argv[4]
LAYER = 'fc6wi'
INDEX = 4

net = caffe.Classifier(MODEL_FILE, PRETRAINED)
net.set_phase_test()
net.set_mode_cpu()
#net.set_mean('data', numpy.load(caffe_root + MEAN_FILE))
net.set_mean('data', numpy.load(MEAN_FILE))
net.set_raw_scale('data', 255)
net.set_channel_swap('data', (2,1,0))

image = caffe.io.load_image(IMAGE_FILE)
net.predict([ image ])
feat = net.blobs[LAYER].data[INDEX].flatten().tolist()
print(' '.join(map(str, feat)))