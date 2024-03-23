## ML Resources

### Getting started with Machine Learning (ML) using Python:
- **Learn Python:** Python is a popular and powerful interpreted language used for both research and development.
- **Understand Machine Learning:** Familiarize yourself with various types of Machine Learning algorithms and when to use them.
- **Install Necessary Libraries:** Download and install `Python SciPy`, the most useful package for machine learning in Python. Other important libraries include `NumPy`, `Pandas`, `Matplotlib`, and `Scikit-learn`.
- **Hands-on Practice:** The best way to learn machine learning is by designing and completing small projects. You can start with simple projects like predicting house prices or classifying emails.
- **Online Courses:** Consider enrolling in online courses. For example, IBM offers a course on Coursera titled "Machine Learning with Python". This course covers topics like linear & non-linear regression, classification techniques using different algorithms, and clustering.
- **Read and Implement Research Papers:** Once you have a good understanding of ML concepts, start reading ML research papers and try to implement them. This will give you a deeper understanding of the field and keep you updated with the latest advancements.
- **Contribute to Open Source Projects:** Contributing to ML open source projects can help you gain practical experience and learn from the community.

### Here are some popular Machine Learning (ML) algorithms:
- **Linear Regression:** A supervised machine learning technique used for predicting and forecasting values that fall within a continuous range.
- **Logistic Regression:** Also known as "logit regression," it's a supervised learning algorithm primarily used for binary classification tasks.
- **Decision Tree:** A type of supervised learning algorithm that is mostly used in classification problems.
- **Support Vector Machines (SVM):** SVMs are a set of supervised learning methods used for classification, regression and outliers detection.
- **Naive Bayes:** A classification technique based on Bayes’ Theorem with an assumption of independence among predictors.
- **K-Nearest Neighbors (KNN):** A type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until function evaluation.
- **Learning Vector Quantization (LVQ):** A prototype-based supervised classification algorithm.
- **Transformers:** A novel architecture that promoted attention mechanisms from ‘piping' in encoder/decoder and recurrent network models to a central transformational technology in their own right.

## Supervised and unsupervised learning
>Supervised and unsupervised learning are two core types of machine learning methods, each with its own unique approach:

**Supervised Learning:**
- Supervised learning involves training a model with labeled datasets to predict outcomes.
- The algorithm learns from the input data that is paired with the correct output labels.
- It aims to find a mapping or relationship between the input variables and the desired output.
- Supervised learning can be divided into two main types:
    - Regression: The goal is to predict a continuous output or value.
    - Classification: The goal is to assign input data to one of several predefined categories or classes.

**Unsupervised Learning:**
- Unsupervised learning uses machine learning algorithms to analyze and cluster unlabeled datasets.
- These algorithms discover hidden patterns in data without the need for human intervention.
- Unsupervised learning models are used for three main tasks: clustering, association, and dimensionality reduction.
- Clustering is a technique for grouping unlabeled data based on their similarities or differences.
  
>In summary, the key difference between supervised and unsupervised learning is that supervised learning uses labeled data to help predict outcomes, while unsupervised learning uncovers patterns in unlabeled data without predefined outcomes.

## Decision Trees  

> Decision Trees are a type of supervised learning algorithm that is mostly used in classification problems. It works for both categorical and continuous input and output variables.

Here's a step-by-step explanation of how Decision Trees work:
- Start at the Root: The decision tree algorithm starts at the root node.
- Choose an Attribute: The algorithm selects an attribute from the dataset and creates branches that correspond to the possible values of that attribute.
- Split the Data: The data is then split into subsets, each corresponding to one of the branches.
- Recursive Splitting: This process is repeated recursively, creating a new decision node for each subset by selecting another attribute and splitting the data again.
- Stop Condition: The recursion stops when either every subset of the data contains only instances of a single class (a pure node), or all the attributes have been used.
- Make Predictions: Once the tree is built, new instances can be classified by navigating through the tree, starting at the root and following the branch that corresponds to the value of each attribute until a leaf node (a final decision) is reached.
- 
Remember, Decision Trees are a simple yet powerful algorithm, and they're the basis for more advanced techniques like Random Forests and Gradient Boosting Machines. They're also easy to understand and interpret, which makes them a popular choice in many machine learning applications.

_For a more detailed look at Decision Trees, you might find these videos helpful:_
- How Decision Tree Works? Beginners Guide
- Decision Tree Classification Clearly Explained!
- Decision Tree 1: how it works
