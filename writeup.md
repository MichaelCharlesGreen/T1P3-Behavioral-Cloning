#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_summary.png "Model Summary"
[image2]: ./examples/center_left_right.png "Center Left Right Images"
[image3]: ./examples/flipped_center_left_right.png "Flipped Images"
[image4]: ./examples/cropped_center_left_right.png "Cropped Images"
[image5]: ./examples/resized_center_left_right.png "Resized Images"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
`python drive.py model.h5`

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model architecture is base on Nvidia's End-to-End model architecture with dropout regularizatioin and ReLU activation functions to introduce non-linearity.

Here is the model summary:

![alt text][image1]

The model crops the image (model.py line 124), resizes the image (model.py line 126), consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 130-137), includes ReLU layers to introduce nonlinearity (model.py lines 133-146) and tanh (model.py line 147), and the data is normalized in the model using a Keras lambda layer (code line 128). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 138, 140, 142). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 111, 113, 114, 153-158). The model was tested by running it through the simulator using `drive.py` and ensuring that the vehicle could stay on the track. Data was recorded and a video of the vehicle driving on the track for one-plus laps was made using `video.py` and saved as `run1.mp4`.

####3. Model parameter tuning

The model used an adam optimizer, and the learning rate was not tuned manually from the default value (model.py line 116, 147, 148). Dropout was set to `0.5` and the number of epochs was `5` (model.py lines 117 and 118).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the Udacity provided training data.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

Instead of driving recovery data from the left and right sides of the lane, I used the left and right camera images with a correction of `0.25` (model.py lines 57-75).

The overall strategy for deriving a model architecture was to start with the simplest model, progress to LeNet and end with Nvidia's End-to-End model architecture. Experimentation led me to this, as well as, the lessons and project demonstration.

In order to gauge how well the model was working, I split my image and steering angle data into training and validation sets prior to augmenting the training set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by adding dropout layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I discovered a mistake in my code. I was reading images in using `cv2` which is `BGR`. Once I converted to `RGB` (model.py line 61), the car was able to successfully navigate one-plus laps.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 120-148) consisted of a convolution neural network with the following layers and layer sizes ...

The model crops the image (model.py line 124), resizes the image (model.py line 126), consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 130-137), includes ReLU layers to introduce nonlinearity (model.py lines 133-146) and tanh (model.py line 147), and the data is normalized in the model using a Keras lambda layer (code line 128).

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I employed many strategies and uploaded (taking hours at times) much data. Some of the tactics I used were as follows. I drove laps fast, I drove laps slow, I drove laps at moderate speed and slowed down (to collect more images) and trouble spots like the bridge, the left turn after the bridge, the only right turn, which is after that left turn and areas where there was a dirt edge to the road.

In the end, I used the Udacity supplied driving data and augmented it by using the left and right camera images with correction (recovery data), flipped images (to balance left and right turn data) (model.py lines 57-89).

I used the left and right camera images with a correction so that the vehicle would learn to recover from the edges of the lane.

These images show what center, left and right camera images look like:

![alt text][image2]

Here are images that has then been flipped:

![alt text][image3]

Here are images that has then been cropped:

![alt text][image4]

Here are images that has then been resized:

![alt text][image5]

The Udacity dataset comprised 8036 rows referencing three camera images and on steering angle data each. Using the left, center, and right camera images and flipping the data resulted in six times as much data.

After the augmentation process, I had 48,216 number of data points. I then preprocessed this data by having the model (taking advantage of the GPU) crop and resize the images (model.py line 122 and 124).

The data set was randomly shuffled (model.py line 45, 95) and returned from the generator.

20% of the data into a validation set (model.py line 111).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.

The number of epochs was `5`. Both the mse and validation error were still going down at this point. But, given that the validation error does not mean that much for the given number of training samples being used and that the model was able to successfully guide the care around the track, I stopped here and did not press my luck.

I used an adam optimizer so that manually training the learning rate wasn't necessary.