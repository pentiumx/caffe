from os import rename, listdir
from os.path import isfile, join
import sys
import os

if len(sys.argv) != 2:
  print 'Please enter a valid argument.'
path = sys.argv[1]
caffe_root = '/home/ispick/Projects/caffe/'

# Get file names
#files = [ f for f in sorted(listdir(path)) if isfile(join(path, f)) ]
#for counter, filename in enumerate(files):
#  rename(filename, 'image_0%d' % counter)

for counter, f in enumerate(sorted(listdir(path))):
  filename, ext = os.path.splitext(os.path.join(path, f))
  if isfile(join(path, f)):
    print os.path.join(caffe_root, path, f)
    print os.path.join(caffe_root, path, 'image%d' % counter, ext)
    rename(os.path.join(caffe_root, path, f), os.path.join(caffe_root, path, 'image'+str(counter)+str(ext)))

# Output file names to a list file
