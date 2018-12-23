# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_image/amount-each-class.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ryutaShitomi/traffic_sign_classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is a diagram visualizing the dataset. This is a histogram. It shows how many images each class has. As you can see from here, there are wide variations in the number of sheets in each class.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...
As a first step, I decided to convert the images to equalize the histogram of lightness because there are many dark images.
I did this for every train, validation, test dataset. This is the normalization I did.
The following shows the original image and the converted image.

<img src='./writeup_image/equalizeLightness.jpg' />

I decided to generate additional data because
I decided to generate additional data because there are variations in the number of images for each class. In order to eliminate this variation, I increased the number of images of that class when the number of data is less than 900 in class.
I increased the data by rotating, expanding and shrinking.

Here is an example of an original image and an augmented image:

<img src='./writeup_image/data_augmentation.jpg' />


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model introduced the Inception Module. In addition, my model includes global average pooling after the Inception Module.
And I put 3 fully connected layers. Also, batch normalization is included in the middle.
My final model consisted of the following layers:

| branch1          | Description              | Main Layer       | Description              | breanch2        | Description            |
|:----------------:|:------------------------:|:----------------:|:------------------------:|:---------------:|:----------------------:|
|                  |                          | Input            | 32x32x3 RGB image				|                 |                        |
| layer1: Inception1(Conv5x5)| 1x1 stride, valid, out:28x28x10|  |                          | layer1: Inception2_1(Conv3x3)| 1x1 stride, valid, out:30x30x10 |
| layer1: RELU     | activation               |                  |                  				| layer1: RELU    | activation             |
| layer1: BN       | Batch Normalization      |                  |                  				| layer1: Inception2_2(Conv3x3)| 1x1 stride, valid, out:30x30x20 |
|         ↓        |             ↓            |                  |                  				| layer1: RELU    | activation             |
|         ↓        |             ↓            |                  |                  				| layer1: BN      | Batch Normalization    |
|                  |                          | layer1: Merge    | Merge inception1 and inception2, out: 28x28x30 |             |      |
| layer2: Inception1(Conv5x5)| 2x2 stride, same, out:14x14x50|   |                  				| layer2: Inception2(Conv3x3)| 2x2 stride, same, out:14x14x50  |
| layer2: RELU     | activation               |                  |                          | layer2: RELU    | activation             |
| layer2: BN       | Batch Normalization      |                  |                          | layer2: BN      | Batch Normalization    |
|                  |                          | layer2: Merge    | Merge inception1 and inception2, out: 14x14x100|             |      |
| layer3: Inception1(Conv5x5)| 1x1 stride, valid, out:10x10x120| |                          | layer3: Inception2_1(Conv3x3)| 1x1 stride, valid, out:12x12x100|
| layer3: RELU     | activation               |                  |                          | layer3: RELU    | activation             |
| layer3: BN       | Batch Normalization      |                  |                          | layer3: Inception2_2(Conv3x3)| 1x1 stride, valid, out:12x12x120|
|         ↓        |             ↓            |                  |                          | layer3: RELU    | activation             |
|         ↓        |             ↓            |                  |                          | layer3: BN      | Batch Normalization    |
|                  |                          | layer3: Merge    | Merge inception1 and inception2, out: 12x12x240|            |       |
|                  |                          | GAP              | Global Average Pooling out:240 |           |                        |
|                  |                          | layer4: Fully Connected  | input: 240, output: 150|           |                        |
|                  |                          | dropout          | dropout keep_prob: 0.75 or 0.5 |           |                        |
|                  |                          | layer5: Fully Connected  | input: 150, output: 100|           |                        |
|                  |                          | dropout          | dropout keep_prob: 0.75 or 0.5 |           |                        |
|                  |                          | output           | input: 100, output: 43         |           |                        |
|                  |                          | Softmax          | input: 43, output: 43          |           |                        |

The graph created with the tensorboard is shown [here](https://github.com/ryutaShitomi/traffic_sign_classifier/blob/master/writeup_image/model.png).


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the my model, I did a stochastic gradient descent method. Since the number of augmentation train data is 53037, I chose the batch size to be the common multiple.And, epochs is 20.
The hyper parameter values are shown below.

* When the accuracy for training data is 94% or less<br>
  learning rate: 0.1<br>
	BATCH_SIZE: 83

* When the accuracy for training data is 95% or greater<br>
  learning rate: 0.03<br>
	BATCH_SIZE: 83*9<br>


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 96.1%
* test set accuracy of 94.9%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* First, I used the LeNet architecture.
* At that time, the accuracy for validation data and training data was very low. So, I thought that model is underfitting. I used an inception module with a 3x3 convolution layer and a 5x5 convolution layer.
* I used batch normalization.
* Using the Inception Module and BN, I achieved 93% accuracy for validation data.
* After that, I used Global Average Pooling based on GoogLeNet. By doing this, I achieved 96% accuracy.
* I used AdaGrad for the optmizer, butt I changed it to GradientDescent because the accuracy was reduced in certain point when I used AdaGrad.
* I inserted dropout layer in the fully connected layers and performed ensemble learning.


##### The difference in feature extraction by the Inception Module is shown below.

<img src='./writeup_image/visualization_inception1_l1.jpg'  />
<img src='./writeup_image/visualization_inception2_l1.jpg' />
<img src='./writeup_image/visualization_inception1_l2.jpg' />
<img src='./writeup_image/visualization_inception2_l2.jpg' />
<img src='./writeup_image/visualization_inception1_l3.jpg' />
<img src='./writeup_image/visualization_inception2_l3.jpg' />

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<img src='./test_images/Ahead only.jpg' width=200 height=200 />
<img src='./test_images/Keep right.jpg' width=200 height=200 />
<img src='./test_images/No entry.jpg' width=200 height=200 /><br><br>
<img src='./test_images/Priority road.jpg' width=200 height=200 />
<img src='./test_images/Right-of-way at the next intersection.jpg' width=200 height=200 />
<img src='./test_images/Roundabout mandatory.jpg' width=200 height=200 />

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
