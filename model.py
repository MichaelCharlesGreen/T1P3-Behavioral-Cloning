# Imports
import csv
import cv2
import numpy as np 
import keras

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.backend import tf as ktf
from keras.models import Sequential, Model
from keras.optimizers import Adam 
from keras.layers import Flatten, Dense, Lambda, Dropout, Input
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab.
import matplotlib.pyplot as plt

'''
While a very low loss does not mean that your car will drive well. It is possible that you over 
trained your model, in which case the model is optimizing loss at the cost of accuracy. Or that 
your learning rate was too high and your loss dropped very steeply, which will also lead to poor 
test time performance.

But if none of these things are true then (along with @Alex_Cui's suggestions) you should double 
check the following things:

Are you pre-processing your images for training (crop, resize, normalize)? 
If yes and if this pre-processing happens outside the model then you should include this 
pre-processing for images drive.py.
drive.py uses PIL to load images. So images are read in the RGB format. 
Are you using cv2 to read images during training? 
If yes you will have to convert them from BGR to RGB.
'''

# generator to solve out-of-memory error.
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			# Collect camera images and their corresponding steering angles from the csv file.
			images = []
			angles = []
			for row in batch_samples:
				# A row from the csv file.
				# There are three string paths to the center, left, and right camera images.
				n_cameras=3
				dir_path = './data/IMG/'
				for image_path_index in range(n_cameras):
					image_path = dir_path+row[image_path_index].split('/')[-1]
					#print(image_path)
					image = cv2.imread(image_path)
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					#img_crop = img[56:150, :, :]
					#img_resize = cv2.resize(img_crop, (200, 66))
					images.append(image)
				    
				# There is a steering angle correction for the left and right camera images.
				# This correction will be used when training the net to steer away from the lane's edge,
				# rather than using the steering angle associated with the center image, which is the
				# angle in the csv file.
				correction=0.25
				#print('row[3]',row[3])
				angle = float(row[3])
				angles.append(angle)
				angles.append(angle+correction) # left image
				angles.append(angle-correction) # right image

			# Collect camera images and their corresponding steering angles from the
			# orginal data and the flipped data.
			augmented_images = []
			augmented_angles = []
			for image, angle in zip(images, angles):
			    augmented_images.append(image)
			    augmented_angles.append(angle)
			    # Flip the image data to simulate CW turns where the track is all CCW turns.
			    # 1 specifies flipping about the vertical axis rather than the horizontal.
			    flipped_image = cv2.flip(image, 1)
			    flipped_angle = angle * -1.0
			    augmented_images.append(flipped_image)
			    augmented_angles.append(flipped_angle)

			# trim image to only see section with road
			X_train = np.array(augmented_images)
			y_train = np.array(augmented_angles)

			yield shuffle(X_train, y_train)

# Used by model to resize the image for Nvidia end-to-end model architecture.
def resize_image(image):
	import tensorflow as tf 
	return tf.image.resize_images(image, [66, 200])

# Collect lines from the csv driving log data file.
samples = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)
samples=samples[1:] # The first line of the csv file is headings, which should not be included.

# Split datasets into training and validation.
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# Generators
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

learning_rate=0.0001 # default
dropout=0.5
n_epochs=5
# Nvidia end-to-end
model = Sequential()
# Crop
model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(160,320,3)))
# Resize
model.add(Lambda(resize_image))
# Normalize and zero-center
model.add(Lambda(lambda x: x/255 - 0.5)) #,input_shape=(66,200,3))
# Convolutional layers; ReLU activation function.
model.add(Convolution2D(24, 5, 5, subsample=(2, 2),
	input_shape=(66, 200, 3),
	activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
#model.add(Dropout(dropout))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
#model.add(Dropout(dropout))
model.add(Flatten())
model.add(Dropout(dropout))
model.add(Dense(1164, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(100, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))
# Adam optimizer; mse loss function
opt = Adam(lr=learning_rate)
model.compile(optimizer=opt, loss='mean_squared_error')

# Fit the data, set aside validation dataset, shuffle, set the number of training epochs.
# Note: given the applied data augmentation, there are now six times as many entries as
# are in the original Udacity dataset.
history_object = model.fit_generator(train_generator,
	samples_per_epoch=6*len(train_samples),
	validation_data=validation_generator,
	nb_val_samples=6*len(validation_samples),
	nb_epoch=n_epochs,
	verbose=1)

# Save the model.
model.save('model.h5')

print(model.summary())