#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

<<<<<<< HEAD
#<<<<<<< HEAD
#EXAMPLE=examples/imagenet
#DATA=data/ilsvrc12
#TOOLS=build/tools

#$TOOLS/compute_image_mean $EXAMPLE/ilsvrc12_train_lmdb \
#  $DATA/imagenet_mean.binaryproto
#=======




=======
>>>>>>> b24babaf57fb3b3d433ab49131b806f622896a37
#./build/tools/compute_image_mean examples/imagenet/ilsvrc12_train_leveldb \
#  data/ilsvrc12/imagenet_mean.binaryproto

./build/tools/compute_image_mean data/ispick/train_db \
  data/ispick/imagenet_mean.binaryproto

echo "Done."
