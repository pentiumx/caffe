#!/usr/bin/env python
import numpy as np
import leveldb
import sys
import caffe
from caffe.io import caffe_pb2
import time
import json

def debug_print (arr):
  print arr
  print arr.size
  for item in arr[0]:
    print item[0]

print(sys.argv[1:][0])
dirname = sys.argv[1:][0]
db = leveldb.LevelDB(dirname)
print type(db.RangeIter())


cnt=0
f = open(dirname + '/feature_values', 'w')
for key, val in db.RangeIter():
  # Get extracted features as a numpy list
  datum = caffe.io.caffe_pb2.Datum()
  datum.ParseFromString(val)
  tmp_list  = caffe.io.datum_to_array(datum)

  # Change the formats
  print tmp_list[0]
  features = []
  for t in tmp_list[0]:
    #print t
    #print type(t)
    features.append(t[0])
  features = map(str, features)
  print type(features)
  print(features)

  # Append it to the file
  print key
  line = { 'index':key, 'value':','.join(features) }
  print line
  f.write(json.dumps(line, ensure_ascii=False) + '\n')
  cnt+=1

f.close()
print 'cnt:' + str(cnt)

