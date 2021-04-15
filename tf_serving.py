import sys
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

print('TensorFlow version: {}'.format(tf.__version__))
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
	               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


MODEL_DIR = "Weights"
def load_data():

	fashion_mnist = keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	# scale the values to 0.0 to 1.0
	train_images = train_images / 255.0
	test_images = test_images / 255.0

	# reshape for feeding into the model
	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
	test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

	
	print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
	print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

	return train_images,train_labels,test_images,test_labels

def get_model():
	model = keras.Sequential([
	  keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, 
	                      strides=2, activation='relu', name='Conv1'),
	  keras.layers.Flatten(),
	  keras.layers.Dense(10, name='Dense')
	])
	model.summary()
	return model

def train():
	
	train_images,train_labels,test_images,test_labels = load_data()
	model = get_model()
	testing = False
	epochs = 1
	model.compile(optimizer='adam', 
	              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              metrics=[keras.metrics.SparseCategoricalAccuracy()])
	model.fit(train_images, train_labels, epochs=epochs)

	test_loss, test_acc = model.evaluate(test_images, test_labels)
	print('\nTest accuracy: {}'.format(test_acc))
	model_save(model)

def model_save(model):

	import tempfile
	
	version = 1
	export_path = os.path.join(MODEL_DIR, str(version))
	print('export_path = {}\n'.format(export_path))

	tf.keras.models.save_model(
	    model,
	    export_path,
	    overwrite=True,
	    include_optimizer=True,
	    save_format=None,
	    signatures=None,
	    options=None
	)
	os.environ['MODEL_DIR'] = MODEL_DIR
	#print('\nSaved model:')

def tf_serving_predict():
	train_images,train_labels,test_images,test_labels = load_data()
	import json
	import requests
	data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
	print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))
	headers = {"content-type": "application/json"}
	json_response = requests.post('http://localhost:8501/v1/models/fashion_model:predict', data=data, headers=headers)
	predictions = json.loads(json_response.text)['predictions']
	for i,pred in enumerate(predictions):
		label = class_names[pred.index(max(pred))] # Matching the prediction with it's correseponding label
		print(label)
		#import pdb;pdb.set_trace()
		plt.imshow(test_images[i]*255)
		plt.show()

	
if __name__ == '__main__':

	if 0:#train
		train()
	if 1:#Predict
		tf_serving_predict()



