from os import listdir
from os.path import isfile, join
import sys

if len(sys.argv) != 2:
  print 'Please enter a valid argument.'
path = sys.argv[1]

# Get file names
files = [ f for f in listdir(path) if isfile(join(path, f)) ]
print len(files)

# Output file names to a list file
f = open(path+'/imagelist','w')
for filename in files:  
  f.write(path+filename+' 0\n') # python will convert \n to os.linesep
f.close() # you can omit in most cases as the destructor will call if

print('done')
