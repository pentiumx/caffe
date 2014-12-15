#!/usr/bin/env python
import numpy as np
import leveldb
import sys
import caffe
from caffe.io import caffe_pb2
import time
import json
import math

filename = sys.argv[1:][0]
print(sys.argv[1:][0])
print(sys.argv[1:][1])
print(sys.argv[1:][2])

# Read the file that contains feature vectors
im0 = int(sys.argv[1:][1])
im1 = int(sys.argv[1:][2])
f0 = None
f1 = None
f = open(filename)
features =  f.readlines()
for counter, feature in enumerate(features):
  tmp  = json.loads(feature)['value']
  #print tmp['value']
  #quit()
  if counter == im0: f0 = tmp
  if counter == im1: f1 = tmp

# Convert string to list
f0 = f0.split(',')
f1 = f1.split(',')
f0 = map(float, f0)
f1 = map(float, f1)

# Compare the two vectors by taking the distanace between them
distance = 0
subtracted  = [b-a for a,b in zip(f0, f1)]
squared  = [s*s for s in subtracted]
for s in squared:
  distance += s
distance = math.sqrt(distance)

print distance

# Finish the process
print 'done'
f.close()
