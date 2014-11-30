#!/usr/bin/env python
import numpy as np
import leveldb
import sys
import caffe
from caffe.io import caffe_pb2
import time

print(sys.argv[1:][0])
dirname = sys.argv[1:][0]
''''''
db = leveldb.LevelDB(dirname)
print type(db.RangeIter())

cnt=0
for key, val in db.RangeIter():
  datum = caffe.io.caffe_pb2.Datum()
  datum.ParseFromString(val)
  arr = caffe.io.datum_to_array(datum)
  print arr
  print arr.size
  for item in arr[0]:
    print item[0]
  cnt+=1
  break

'''h = leveldb.LevelDB(dirname)
datum = caffe_pb2.Datum()
for key_val,ser_str in h.RangeIter():
  datum.ParseFromString(ser_str)
  rows = datum.height;
  cols = datum.width;
  print type(datum.data)
  print len(datum.data)
  img_pre = np.fromstring(datum.data, dtype=np.uint8)
  print type(img_pre)
  img = img_pre.reshape(rows, cols)
  print "\nKey val: ", key_val
  print "Image: ", img
'''
print cnt