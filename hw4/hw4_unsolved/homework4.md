---
title: "ECS171 Homework 4"
---

In this assignment, we will be building a Naïve Bayes classifier and a SVM model for the productivity satisfaction of [the given dataset](https://archive.ics.uci.edu/ml/datasets/Productivity+Prediction+of+Garment+Employees), the productivity of garment employees.

You can find the template for this homework [here](TODO), and you should submit your work to Gradescope [here](TODO).

# Background

### Background for Questions 1-3

The Garment Industry is one of the key examples of industrial globalization in this modern era. It is a highly labor-intensive industry with many manual processes. Satisfying the huge global demand for garment products depends mostly on the production and delivery performance of employees in garment manufacturing companies. Therefore, it is highly desirable for decision makers in the garment industry to track, analyze and predict the productivity performance of working teams in their factories.


#### Equivalence Relation

Let's say if we have two bitstrings, $A = \{a_{15}, a_{14}, a_{13}, ..., a_0\}$ and $B = \{b_{15}, b_{14}, b_{13}, ..., b_0\}$,  We can flip one bit $a_i$ in $A$ to get another bitstring $A'$, such that the difference between $A$ and $A'$ is only one bit. We define the above transformation to be $A \rightarrow A'$.  

We call two bitstrings $A$ and $B$ to be **equivalent** ($A \sim B$) if there exists a sequence $A \rightarrow C_1 \rightarrow C_2 \rightarrow ... \rightarrow C_n \rightarrow B$, where $\forall i, C_i$ belongs to the dataset.

It can be seen that equivalence is both *commutative* ($A \sim B$ iff $B \sim A$) as well as *transitive* ($A \sim B, B \sim C$ implies $A \sim C$).  

We can say that the elements in the above sequence $\{ A, C_1, ..., C_n, B \}$ form an _equivalence class_. Given a new bitstring $X$, we can see that if $X \sim C_i$, $1 \leq i \leq n$, then $X$ will be added to the above equivalence class, and by the transitive property of equivalence relations, $X \sim A$, and $X \sim B$.

##### Example

Let's say we have 4 bitstrings, each of them is 4 bits long. They are *0000, 0010, 0110, 1100*, respectively.  
We can say *0000 ~ 0110* because *0000 -\> 0010 -\> 0110*.  
However, *0000 ≁ 1100*. There may be sequences like *0000 -\> 1000 -\> 1100* or *0000 -\> 0100 -\> 1100*, but neither *1000* nor *0100* is in our dataset.  
Ultimately, *{0000, 0010, 0110}* forms an equivalence class, whereas *{1100}* is the other. As a result, there are two classes.



### Background for Questions 4-5

For a bitstring $S$ in this dataset, we describe $S = \{s_{15}, s_{14}, s_{13}, s_{12}, ..., s_0\}$, where $s_{15}$ is often known as the most significant bit (MSB) and $s_0$ as the least significant bit (LSB). There are duplicated bitstrings in this dataset, but they will not affect this assignment. Don't worry about them.


### Dataset Attribute Information

| Attribute                 | Description                                                                                                   |
|---------------------------|---------------------------------------------------------------------------------------------------------------|
| **date**                  | Date in MM-DD-YYYY                                                                                            |
| **day**                   | Day of the Week                                                                                               |
| **quarter**               | A portion of the month. A month was divided into four quarters                                                |
| **department**            | Associated department with the instance                                                                       |
| **team_no**               | Associated team number with the instance                                                                      |
| **no_of_workers**         | Number of workers in each team                                                                                |
| **no_of_style_change**    | Number of changes in the style of a particular product                                                        |
| **targeted_productivity** | Targeted productivity set by the Authority for each team for each day                                         |
| **smv**                   | Standard Minute Value, it is the allocated time for a task                                                    |
| **wip**                   | Work in progress. Includes the number of unfinished items for products                                        |
| **over_time**             | Represents the amount of overtime by each team in minutes                                                     |
| **incentive**             | Represents the amount of financial incentive (in BDT) that enables or motivates a particular course of action |
| **idle_time**             | The amount of time when the production was interrupted due to several reasons                                 |
| **idle_men**              | The number of workers who were idle due to production interruption                                            |
| **actual_productivity**   | The actual % of productivity that was delivered by the workers. It ranges from 0-1.                           |

### Libraries that can be used: 
`numpy`, `scipy`, `pandas`, `scikit-learn`, `cvxpy`, `imblearn` 

Any libraries used in the discussion materials are also allowed.

### Other Notes

- Don't worry about not being able to achieve high accuracy, it is neither the goal nor the grading standard of this assignment.
- If not specified, you are not required to do hyperparameter tuning, but feel free to do so if you'd like.

### Trouble Shooting

In case you have trouble installing and using `imbalanced-learn` (`imblearn`),  

Run the code cell below, then go to the selection bar at top: Kernel \> Restart.  
Then try `import imblearn` to see if things work.

We are also working on a list of 1200 bitstrings, where each of them
contains 16 bits.  
We will apply K-means Clustering and PCA to this dataset for Q4-5.

# Exercises
## Exercise 1: General Data Preprocessing (10 points)

Our dataset needs cleaning before building any models. Some of the cleaning tasks are common in general, but depending on what kind of models we are building, sometimes we have to do additional processing. These additional tasks will be mentioned in each of the remaining two exercises later.

Note that **we will be using this processed data from Exercise 1 in each of the remaining two exercises**.

For convenience, here are the attributes that we would treat as **categorical attributes**: `day`, `quarter`, `department`, and `team`.

1. Drop the column `date`.
2. For each of the categorical attributes, **print** all the unique elements.
3. For each of the categorical attributes, remap the duplicated items, if you find there are typos or spaces among the duplicated items.
   - For example, "a" and "a " should be the same, so we need to update "a " to be "a".
   - Another example, "apple" and "appel" should be the same, so you should update "appel" to be "apple".
4. Create another column named `satisfied` that records the productivity performance. The behavior is defined as follows. **This is the dependent variable we'd like to classify in this assignment.**
   - Return True or 1 if `actual_productivity` is equal to or greater than `targeted_productivity`. Otherwise, return False or 0, which means the team fails to meet the expected performance.
5. Drop the columns `actual_productivity` and `targeted_productivity`.
6. Find and **print** which columns/attributes have empty values, e.g., `NA`, `NaN`, `null`, `None`.
7. Fill the empty values with 0.

## Exercise 2: Naïve Bayes Classifier (20 points in total)

### Exercise 2.1: Additional Data Preprocessing (5 points)

To build a Naïve Bayes Classifier, we need to further encode our categorical variables.

1. For each of the **categorical attributes**, encode the set of categories using integers $0, 1, 2, ..., n-1$.
   - For example, `["paris", "paris", "tokyo", "amsterdam"]` should be encoded as `[1, 1, 2, 0]`.
   - Note that the order does not really matter, i.e., `[0, 0, 1, 2]` also works. But you have to start with 0 in your encodings.
   - You can find information about this encoding in the discussion materials.
2. Split the data into training and testing sets with a ratio of 80:20.

### Exercise 2.2: Naïve Bayes Classifier for Categorical Attributes (15 points)

Use the categorical attributes **only**, please build a Categorical Naïve Bayes classifier that predicts the column `satisfied`. Report the **testing result** using `classification_report`.

## Exercise 3: SVM Classifier (30 points in total)

### Exercise 3.1: Additional Data Preprocessing (5 points)

To build a SVM Classifier, we need a different encoding for our categorical variables.

1. For each of the **categorical attributes**, encode them with **one-hot encoding**.
   - You can find information about this encoding in the discussion materials.
2. Split the data into training and testing sets with a ratio of 80:20.

### Exercise 3.2: SVM with Different Kernels (15 points)

Using all the attributes we have, please build a SVM that predicts the column `satisfied`.  
Specifically, please:

1. Build one SVM with a **linear kernel**.
2. Build another SVM with an **RBF kernel**.
3. Report the **testing results** of **both models** using `classification_report`.

The kernel is the only required hyperparameter setting. Other hyperparameter tuning is optional. _If_ you choose to tune other hyperparameters, make sure they are identical between the two SVMs. In other words, the kernel setting should be the only difference between the two SVMs.

**Remember to scale your data. The scaling method is up to you.**

### Exercise 3.3: SVM with Over-sampling (10 points)

1. For the column `satisfied` in the **training set**, **print** the frequency of each class.
2. Oversample the **training data**.
3. For the column `satisfied` in the oversampled data, **print** the frequency of each class again.
4. Re-build the two SVMs using the same settings as in Exercise 3.2, but **use the oversampled training data** instead.
   - Do not forget to scale the data first. As always, the scaling method is up to you.
5. Report the **testing results** using `classification_report`.

You can use any over-sampling method listed [here](https://imbalanced-learn.org/stable/references/over_sampling.html#), such as `RandomOverSampler` or `SMOTE`. You are welcome to build your own over-sampler as well. Note that you only need to over-sample the training data, not the testing data.

## Exercise 4: K-Means Clustering (25 points in total)

### Exercise 4.1: K-Means Clustering for Equivalence Classes (10 points)

1. Cluster the dataset with K-Means, using `k=60`.
2. Calculate and report the Akaike Information Criterion.
3. Show the frequency (number of members) of each cluster. Again, you are encouraged to create a bar chart, but printing the numbers is also fine.

### Exercise 4.2: Difference between Agglomerative Clustering and K-Means Clustering (5 points)

1. Explain what are the differences between Agglomerative Clustering and K-Means Clustering.
2. Explain why there is such a difference.

### Exercise 4.3: Generate 2 Clusters (10 points)

1. Re-do the K-Means clustering on our dataset again, but this time we only consider `k=2`.
2. Calculate and report the Akaike Information Criterion.
3. Show the frequency (number of members) of each cluster.
4. Comment on whether this is a better model compared to Exercise 4.1.

## Exercise 5: Principal Component Analysis (15 points in total)

1. Retrieve the projected dataset with PCA, using `n_components=2`.
2. Generate a scatter plot to visualize the projected points, where they should be colored differently based on the assigned cluster in Exercise 4.1.
3. In the first principal component, **print** the weights of all features.
4. Report which feature has the **highest positive** weight in the first principal component.
