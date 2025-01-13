# Assignment 1

In this assignment, Exercises 1-3 explore a car dataset and analyze fuel efficiency. Exercise 4 uses a diabetes dataset.

You will perform exploratory analysis with visualizations, build a Simple Linear Regression model, and build a Polynomial Regression model.

**The dataset has been pre-cleaned and modified** from the original [Auto MPG dataset](https://archive.ics.uci.edu/ml/datasets/auto+mpg).

## Dataset Attributes

1. `mpg`: Miles per gallon (primary fuel efficiency measurement)
2. `displacement`: Cylinder volume in cubic inches 
3. `horsepower`: Engine power
4. `weight`: Weight in pounds
5. `acceleration`: Time in seconds from 0 to 60mph
6. `origin`: Region of origin

### Libraries
1. `numpy`
2. `pandas` 
3. `scikit-learn`
4. `seaborn`
5. `plotly`
6. `matplotlib`

Any libraries used in discussion materials are also allowed.

### Notes
1. Model accuracy is not the primary grading criterion for this assignment
2. Hyperparameter tuning is optional unless specified
3. Reference discussion materials for guidance

## Exercise 1: Exploratory Analysis (20 points)

### Exercise 1.1: Correlation Matrix (10 points)

Generate a Pearson [correlation matrix plot](https://heartbeat.fritz.ai/seaborn-heatmaps-13-ways-to-customize-correlation-matrix-visualizations-f1c49c816f07) as a heatmap.

After generating the plot, explain which attribute would be least suitable as the independent variable for predicting `mpg` using Simple Linear Regression ($y = ax + b$). Justify your answer.

Requirements:
1. Drop the `origin` column when computing correlations
2. Display correlation values on the plot
3. Use a diverging color scale with range [-1, 1] centered at 0

### Exercise 1.2: Pairplot (10 points)

Generate a pairplot (scatter plot matrix) of the dataset.

After generating the plot, determine which method (Linear Regression, Polynomial Regression or Logistic Regression) would best predict `mpg` using `horsepower`. Justify your choice.

Requirements:
1. Color points based on `origin`
2. Diagonal plots are optional but `origin`-based distributions are recommended

## Exercise 2: Linear and Polynomial Regression (30 points)

### Exercise 2.1: Data Splitting (5 points)

Split data into 80% training and 20% testing sets.

### Exercise 2.2: Simple Linear Regression (10 points)

Build a Simple Linear Regression model to predict `mpg` using one attribute of your choice (excluding `origin`).

Requirements:
1. Report testing MSE

### Exercise 2.3: Polynomial Regression (15 points)

Build polynomial regression models of degrees 2-4 to predict `mpg` using the same attribute from Exercise 2.2.

Analyze whether the three models show signs of overfitting based on their errors. Justify your reasoning.

Requirements:
1. Report training MSE for each degree
2. Report testing MSE for each degree

## Exercise 3: Overfitting and Underfitting (25 points)

Using the provided fitting dataset containing actual train/test data and three rotations:

### Exercise 3.1: SSE Calculation

Calculate SSE for the three predictions using the actual data for train and test from the fitting dataset.

Requirements:
1. Show SSE calculations
2. Highlight values for all predictions and actual data

### Exercise 3.2: Model Classification

Classify the predictions into:
1. Base prediction
2. Overfitting prediction  
3. Underfitting prediction

There is one of each.

Provide justification for each classification. 

## Exercise 4: Outlier Detection (25 points)

Using the provided diabetes dataset:

### Exercise 4.1: Box Plot Analysis

Create a box plot of the `BloodPressure` attribute.

Requirements:
1. Highlight outliers with distinct colors

### Exercise 4.2: Anomaly Detection

Analyze the `BMI` and `Insulin` features using One-Class SVM for anomaly detection.

Requirements:
1. Create a scatter plot similar to Lecture 2 Slide 11
2. Annotate outlier points
