#!/usr/bin/env python
import numpy as np
import leveldb
import sys
import caffe
from caffe.io import caffe_pb2
import time
import json

dirname = sys.argv[1:][0]
db = leveldb.LevelDB(dirname)

cnt=0
f = open(dirname + '/feature_values', 'w')
for key, val in db.RangeIter():
  # Get extracted features as a numpy list
  datum = caffe.io.caffe_pb2.Datum()
  datum.ParseFromString(val)
  tmp_list  = caffe.io.datum_to_array(datum)

  # Change the formats
  features = []
  for t in tmp_list[0]:
    features.append(t[0])
  print len(features)
  features = map(str, features)

  # Append it to the file
  line = { 'index':key, 'value':','.join(features) }
  json_string = json.dumps(line, ensure_ascii=False)
  #print json_string
  #print ','.join(features)
  f.write(json_string + '\n')
  cnt+=1

f.close()


