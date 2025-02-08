---
title: "ECS171 Homework 3"
---

For this assignment, you will be developing an artificial neural network to classify data given in the [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) dataset. This data set was obtained as a part of a research study by University of Minho, Portugal, where the quality of multiple different wines is classified on a scale of 1-10. More details on the study can be found in the following [research paper](https://repositorium.sdum.uminho.pt/bitstream/1822/10029/1/wine5.pdf).

Please submit this homework through Gradescope from [here](https://www.gradescope.com/courses/704922/assignments/4042246). You could find the homework dataset and template [here](https://canvas.ucdavis.edu/courses/874510/files/22926006).

### About the Data Set

Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. We are using the white wine data set that provided in the zip file for your. The goal is to model wine quality based on physicochemical tests [[Cortez et al., 2009]](http://www3.dsi.uminho.pt/pcortez/wine/).

This datasets is related to red variants of the Portuguese "Vinho Verde" wine.The dataset describes the amount of various chemicals present in wine and their effect on it's quality. The datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are much more normal wines than excellent or poor ones).Your task is to predict the quality of wine using the given data. **For this assignment, we are consider this as a classification problem where you will classify the data into 1-10 different level of quality.**

#### Number of Instances (records in the data set): **4898**

#### Number of Attributes (fields within each record, including the class): **12**

#### Data Set Attribute Information:

#### **Input variables (based on physicochemical tests):**

1. fixed acidity
2. volatile acidity  
3. citric acid
4. residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10. sulphates
11. alcohol

**Output variable (based on sensory data):**

1. quality (score between 1 and 10)

#### Libraries that can be used:

1. `numpy`, `scipy`, `pandas`, `sklearn`, `tensorflow`, `keras`, `seaborn`, `plotly`, `matplotlib`
2. Any other library are fine to use if you could finish the task correctly.

#### Other Notes

1. Don't worry about getting a very high accuracy
2. Discussion materials should be helpful for doing the assignments.
3. The homework submission should be a `.ipynb` file.

### Exercise 1 - General Data Preprocessing (10 points)

Follow what's have done in Homework 2 Exercise 1 and preprocess the data for this homework.

1. As the classes are categorical, use one-hot encoding to represent the set of classes. You will find this useful when developing the output layer of the neural network.
   - **Note that the dataset doesn't include data points from all grading levels, but you may still need to create the one-hot encoding by 10 cols.**
2. Normalize each field of the input data using the min-max normalization technique.

### Exercise 2 - Neural Network and Gradient (20 points)

Suppose there is a neural network with a 2-3-2-1 structure. The nodes in the first hidden layer (from left to right) use a linear activation function while the second hidden layer neurons and the output neuron use a sigmoid activation function. The output $\hat{y}$ is the prediction of the network. The symbols $h$ and $\hat{y}$ are the values after applying the activation fuction.

Loss is calculated by $L = \frac{1}{2} (y - \hat{y})^2$

After a Feed-Forward Pass, write the updated weights during backward pass for $w_{ko}$, $w_{jk}$, and $w_{ij}$ with a learning rate of $\lambda$ (You just need to write 3 formula for those 3 stage of weight in general). You have to show how you compute ${\nabla}w$ for each weight mathematically.

Make sure that you show all your work. Refer to Lecture 8.

![Neural Network Diagram](./Homework_3/ex2.png)

You may use the notations provided below.

$$z_{j} = {\sum}w_{ij}x_{i} + b_{j}; x_{j} = z_{j}$$
$$z_{k} = {\sum}w_{jk}x_{j} + b_{x}; x_{k} = {\sigma}(z_{k})$$
$$z_{o} = {\sum}w_{ko}x_{k} + b_{o}; ŷ = {\sigma}(z_{o})$$

### Exercise 3 : Building a Feed-Forward Neural Network(40 points in total)

#### Training and Testing the Neural Network (40 points)

Design a 5-layer artificial neural network, specifically a feed-forward multi-layer perceptron (using the sigmoid activation function), to classify the type of 'quality' given the other attributes in the data set, similar to the one mentioned in the paper above.

For training and testing the model, split the data into training and testing set by **90:10** and use the training set for training the model and the test set to evaluate the model performance.

Consider the following hyperparameters while developing your model:

1. Number of nodes in each hidden layer should be (23, 17, 13)
2. Learning rate should be 0.4
3. Number of epochs should be 500
4. The sigmoid function should be used as the activation function in each layer
5. Stochastic Gradient Descent should be used to minimize the error rate

**Requirements once the model has been trained:**

1. A confusion matrix for all classes(should be 10 in this case), where you need to specifying what is the true positive, true negative, false positive, and false negative cases for each category in the class
2. The mean squared error (MSE) of the model

**Notes:**

1. Splitting of the dataset should be done **after** the data preprocessing step.
2. The mean squared error (MSE) values obtained **should be positive**

### Exercise 4 : k-fold Cross Validation (10 points in total)

In order to avoid using biased models, use 10-fold cross validation to generalize the model based on the given data set.

**Requirements:**

1. The MSE values during each iteration of the cross validation
2. The overall average MSE value

**Note:** The mean squared error (MSE) values obtained should be positive.

### Exercise 5 : Hyperparameter Tuning (20 points)

Use either grid search or random search methodology to find the optimal number of nodes required in each hidden layer, as well as the optimal learning rate and the number of epochs, such that the MSE of the model is minimum for the given wine quality.

**Requirements:**

1. The set of optimal hyperparameters
2. The mimimum MSE achieved using this set of optimal hyperparameters

**Note:** Hyperparameter tuning takes a lot of time to execute. Make sure that you choose the appropriate number of each hyperparameter (at least 3 of each), and that you allocate enough time to execute your code.
