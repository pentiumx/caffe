import caffe
import numpy as np
import matplotlib.pyplot as plt

net_path   = "models/ispick_finetune/deploy.prototxt"
model_path = "models/ispick_finetune/finetune_sketch_style_iter_10000.caffemodel"
mean_path  = "data/ilsvrc12/imagenet_mean.npy"
#IMAGE_FILE = '/home/ispick/Projects/illust_detection/detect_natural_images/sketch/airplane/1.png'
image_dir='/home/ispick/Projects/caffe/debugimages/'
image_files = [
	'/home/ispick/Projects/illust_detection/detect_natural_images/sketch/airplane/1.png',
	image_dir+'madoka_sketch0.png',
	image_dir+'madoka_sketch1.jpg',
	image_dir+'madoka_illust.jpg',
	image_dir+'madoka_real.jpg',
]

print np.load(mean_path).shape


net = caffe.Classifier(
    net_path, model_path, mean=np.load(mean_path),
    channel_swap=(2, 1, 0), raw_scale=255,
    image_dims=(256, 256))


for image_file in image_files:
	input_image = caffe.io.load_image(image_file)
	prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
	#print 'prediction shape:', prediction[0].shape
	print 'predicted class:', prediction[0].argmax()
	


"""
input_image = caffe.io.load_image(IMAGE_FILE)
plt.imshow(input_image)
prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
print 'prediction shape:', prediction[0].shape
plt.plot(prediction[0])
print 'predicted class:', prediction[0].argmax()
plt.show()
"""