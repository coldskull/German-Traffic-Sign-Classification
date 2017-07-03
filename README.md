# German Traffic Sign Recognition 

---


The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/histogram.png "histogram"
[image2]: ./examples/random_sample_images.png "Random sample images"
[image3]: ./examples/11.png "Orginal"
[image4]: ./examples/12.png "Blurred and saturated"
[image5]: ./examples/13.png "splotches 1"
[image6]: ./examples/15.png "splotches 2"
[image7]: ./examples/14.png "Blurred and Eroded"
[image8]: ./examples/16.png "Scaled and Rotated"
[image9]: ./examples/splotches.png "splotches approach"
[image10]: ./examples/inputs.png "input images"
[image11]: ./examples/softmax.png "softmax of outputs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/coldskull/German-Traffic-Sign-Classification)

## Data Set Summary & Exploration


#### 1. A basic summary of the data set.

The summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

The bar chart below shows the frequency distribution of the different classes of images in the training set.

![alt text][image1]





Here are some randomly selected images from the training set.

![alt text][image2]

## Design of the Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to generate additional data because I wanted my network to become more agnostic to blurring/scaling/rotations/obstructions.  
I used various techniques for augmenting the training image set. The main techniques used were:

1. Blurred and saturated images. In order to mimic real life condition where some images maybe overexposed and blurred. I used gaussian blur with kernel size of 7x7 then added the resulting image to the original. Here is a sample of the orginal and augmented image:

![alt text][image3]
*Original*
![alt text][image4]
*Blurred and Saturated*

2. Random blotches added to the image. This technique tries to mimic real life scenarios where part of the sign maybe obscured or covered by stickers/graffiti. The splotches themselves have a random color so as not to make the network depend on these. These images really helped the network generalize better. See more info below.

![alt text][image3]
*Original*
![alt text][image5]
*Random splotches*
![alt text][image6]
*Random splotches*

3. Blurred and eroded images. Here images are blurred using gaussian blur (7x7) followed by opencv 'erode' operation

![alt text][image3]
*Original*
![alt text][image7]
*Blurred/Eroded*


4. Random rotation and scaling. For this specific augemntation, the original images were rotated randomly by 15-40 degree and scaled randomly by 0.5-1.6. This would make the network more agnostic to scaling and minor rotation (which can happen in real life)

![alt text][image3]
*Original*
![alt text][image8]
*Scaled/Rotated*

Before feeding the augmented training set to the network, I normalized the pixel values (-127 to +127) so that no node in the network 'saturates'. I saw some minor difference in performace.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers (based on LeNet with some modifications):

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x10				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU          		|         									|
| Max Pooling			| 2x2 stride, outputs 5x5x16 (Flattened output=400)     									|
| FC Layer		        | input 400, output 200											|
| RELU   				|												|
| FC Layer   			| input	200, ouput 84										|
| RELU   				|												|
| FC Layer   			| input 84, output 43											|
| Logits   				|												|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used an AWS g2.2xlarge instance to train the model. I played around with pretty much all hyperparameters. The final set used for submission was:

EPOCHS = 24

BATCH_SIZE = 124

learning_rate = 0.0008

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.953
* validation set accuracy of 0.937
* test set accuracy of 0.937

An iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    
    The first architecture chosen was based on LetNet arch used for MNIST (with modifications for 3 channel input image). Since the problem domain is similar,i.e., classifying a set of images, I started with this architecture as a base.

* What were some problems with the initial architecture?
    
    Initial architecture had a low accuracy on validation set (0.86)
    
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

    Besides changing the input layer to take in 3 channel images, I increased the depth of the convolution layers to 10 and 16 respectively. This gave better performance. I increased the depth because the traffic sign data is semantically more complex than MNIST data for digit classification.
I also incresed the width of fully connected layers which gave better accuracy.
I did not use pooling layers since there was no significant overfitting occuring (i belive this was because of the 'splotches' approach in data augmentation)
See results and discussion on this [thread](https://discussions.udacity.com/t/is-such-a-train-validation-accuracy-curve-considered-a-good-one/281251/16)

![alt text][image9]

* Which parameters were tuned? How were they adjusted and why?

    I also realized that BATCH_SIZE affects the results drastically in practice. I learnt this the hard way. I had initially set the BATCH_SIZE to 1024 since i was using a GPU and was getting terrible results (0.89). Just changing BATCH_SIZE to 128 gave be better results (0.94)
See this [thread](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network) and [paper](https://arxiv.org/abs/1609.04836) 
The batch_size and EPOCHS were tuned to get highest accuracy on validation set. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

    One important choice i made is to avoid a drop layer since my network was generalizing well on validation and test set


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image10] 

The image quality was pretty low, espically for the "slippery road" image. Also, for the speed limit sign, the '5' is partilly visible.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead Only      		| Ahead Only   									| 
| (50km/h)  | (30km/h) 										|
| End of all speed and passing limits					| End of all speed and passing limitseld											|
| Roundabout mandatory	      		| Roundabout mandatory					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.937 considering the quality of images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.
Here are the softmax probabilities for all 5 images. Although the network produces a wrong result for the speed limit sign, the correct answer was 2nd in line.

![alt text][image11] 
