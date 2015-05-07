import numpy as np
import matplotlib.pyplot as plt
import math
#%matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
#IMAGE_FILE = 'images/cat.jpg'
IMAGE_FILE = '/usr/prakt/p041/OzgeAkin/MachineLearningPraktikum/week2/ex2_images/IMG_1.jpg'

import os
if not os.path.isfile(PRETRAINED):
    print("Downloading pre-trained CaffeNet model...")
    #!../scripts/download_model_binary.py ../models/bvlc_reference_caffenet
    
caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
                       

input_image = caffe.io.load_image(IMAGE_FILE)
#plt.imshow(input_image)

prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
print 'prediction shape:', prediction[0].shape
#plt.plot(prediction[0])
print 'predicted class:', prediction[0].argmax()  

prediction = net.predict([input_image], oversample=False)
print 'prediction shape:', prediction[0].shape
#plt.plot(prediction[0])
predicted_class = prediction[0].argmax()
print 'predicted class:', predicted_class
print prediction[0][predicted_class]
#%timeit net.predict([input_image]) 
# Resize the image to the standard (256, 256) and oversample net input sized crops.
input_oversampled = caffe.io.oversample([caffe.io.resize_image(input_image, net.image_dims)], net.crop_dims)
# 'data' is the input blob name in the model definition, so we preprocess for that input.
caffe_input = np.asarray([net.transformer.preprocess('data', in_) for in_ in input_oversampled])
# forward() takes keyword args for the input blobs with preprocessed input arrays.
#%timeit net.forward(data=caffe_input)

caffe.set_mode_gpu()
prediction = net.predict([input_image])
print 'prediction shape:', prediction[0].shape
entropy = 0
for i in range(1000):
	entropy = prediction[0][i]*math.log(prediction[0][i],2)

print 'entropy: ', entropy

#plt.plot(prediction[0])
# Full pipeline timing.
#%timeit net.predict([input_image])
# Forward pass timing.
#%timeit net.forward(data=caffe_input)

  
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
try:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
    #!../data/ilsvrc12/get_ilsvrc_aux.sh
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print labels[top_k]
# sort top k predictions from softmax output
#top_k = net.blobs['prob'].data[4].flatten().argsort()[-1:-6:-1]
top_k = net.blobs['prob'].data[4][predicted_class]
#print labels[top_k]
print top_k
  
out = net.forward()
print out['prob'][0][predicted_class]
