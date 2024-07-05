# Plant Leaf Classification and Clustering 🍀

This project focuses on the classification and clustering of plant leaves using various machine learning models. The primary steps involved are data preprocessing, hyperparameter tuning using GridSearchCV, and evaluating the performance of different classification and clustering algorithms.


## Introduction
This project aims to classify and cluster plant leaves using various machine learning techniques. The classification models include:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gaussian Naive Bayes
- Decision Tree
- Random Forest
- AdaBoost
- Multi-Layer Perceptron (MLP)

For clustering, the models used are:
- K-Means
- Agglomerative Clustering
- DBSCAN
- OPTICS
- Gaussian Mixture



  ## Preprocessing
Before applying the models, the data is preprocessed as follows:
- **Normalization:** StandardScaler is used to normalize the data due to the importance of distance metrics in clustering and classification.
- **Dimensionality Reduction:** PCA is used to reduce the dimensionality of the data to improve model performance and reduce computation time.


## Classification Models
### K-Nearest Neighbors (KNN)
- **Description:** A simple, non-parametric method that assigns labels based on the k-nearest neighbors.
- **Parameters:**
  - `n_neighbors`: Number of neighbors to use.
  - `metric`: Distance metric to use.


### Support Vector Machine (SVM)
- **Description:** Finds the optimal hyperplane that maximizes the margin between classes.
- **Parameters:**
  - `C`: Regularization parameter.
  - `kernel`: Type of kernel to use (linear, poly, rbf, sigmoid).
  - `gamma`: Kernel coefficient.
  - `degree`: Degree of the polynomial kernel function.

 
  ### Gaussian Naive Bayes
- **Description:** Assumes Gaussian distribution for each feature and independence between features.
- **Parameters:**
  - `var_smoothing`: Portion of the largest variance of all features added to variances for stability.


  ### Decision Tree
- **Description:** Splits data into branches to form a tree based on feature values.
- **Parameters:**
  - `criterion`: Function to measure the quality of a split (gini, entropy).
  - `max_depth`: Maximum depth of the tree.
  - `min_samples_split`: Minimum number of samples required to split an internal node.
  - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.


### Random Forest
- **Description:** An ensemble method that fits multiple decision trees on various sub-samples.
- **Parameters:**
  - `n_estimators`: Number of trees in the forest.
  - `max_depth`: Maximum depth of the trees.
  - `min_samples_split`: Minimum number of samples required to split an internal node.
  - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
  - `max_features`: Number of features to consider when looking for the best split.
 

  ### AdaBoost
- **Description:** Boosting method that combines weak classifiers to form a strong classifier.
- **Parameters:**
  - `n_estimators`: Number of weak classifiers to combine.
  - `learning_rate`: Weight applied to each classifier at each iteration.
  - `base_estimator`: The base estimator from which the boosted ensemble is built.


### Multi-Layer Perceptron (MLP)
- **Description:** A feedforward artificial neural network model.
- **Parameters:**
  - `hidden_layer_sizes`: Number of neurons in each hidden layer.
  - `activation`: Activation function for the hidden layer.
  - `solver`: The solver for weight optimization.
  - `learning_rate`: Learning rate schedule for weight updates.
  - `max_iter`: Maximum number of iterations.



## Clustering Models


### K-Means
- **Description:** Partitions data into K clusters by minimizing the variance within each cluster.
- **Parameters:**
  - `n_clusters`: Number of clusters.
  - `init`: Method for initialization (k-means++ or random).
  - `n_init`: Number of time the k-means algorithm will be run with different centroid seeds.



### Agglomerative Clustering
- **Description:** Hierarchical clustering that merges pairs of clusters.
- **Parameters:**
  - `n_clusters`: Number of clusters to find.
  - `linkage`: Linkage criterion to use (ward, complete, average, single).
  - `affinity`: Metric used to compute the linkage (euclidean, l1, l2, manhattan, cosine).



### DBSCAN
- **Description:** Density-based clustering algorithm.
- **Parameters:**
  - `eps`: Maximum distance between two samples for one to be considered as in the neighborhood of the other.
  - `min_samples`: Number of samples in a neighborhood for a point to be considered as a core point.


### OPTICS
- **Description:** Orders points to identify the clustering structure.
- **Parameters:**
  - `eps`: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
  - `min_samples`: Minimum number of samples in a neighborhood for a point to be considered as a core point.
  - `xi`: Minimum steepness on the reachability plot that constitutes a cluster boundary.


### Gaussian Mixture
- **Description:** Models data using a mixture of Gaussian distributions.
- **Parameters:**
  - `n_components`: Number of mixture components.
  - `covariance_type`: Type of covariance parameters to use (full, tied, diag, spherical).



## Performance Analysis
The performance of each classification model was evaluated using various metrics such as accuracy, precision, recall, and F1-score. MLP classifier showed the highest performance due to its ability to capture complex patterns in the data. For clustering, the elbow method was used to determine the optimal number of clusters for K-Means, and other models were evaluated based on their ability to handle noisy data and varying cluster shapes.



### Classifier Performance

| Classifier              | Accuracy | Precision | Recall   | F1 Score |
|-------------------------|----------|-----------|----------|----------|
| MLP                     | 0.941176 | 0.964461  | 0.941176 | 0.942577 |
| Random Forest           | 0.882353 | 0.928922  | 0.882353 | 0.886275 |
| SVM                     | 0.882353 | 0.922059  | 0.882353 | 0.881559 |
| KNN                     | 0.867647 | 0.922059  | 0.867647 | 0.874767 |
| AdaBoost                | 0.823529 | 0.887255  | 0.823529 | 0.831956 |
| Gaussian Naive Bayes    | 0.808824 | 0.862745  | 0.808824 | 0.805952 |
| Decision Tree           | 0.647059 | 0.658824  | 0.647059 | 0.615920 |



### Clustering Performance

| Clustering Algorithm | Silhouette Score |
|----------------------|------------------|
| DBSCAN               | 0.516671         |
| OPTICS               | 0.347394         |
| KMeans               | 0.327922         |
| Agglomerative        | 0.316300         |
| GMM                  | 0.256321         |
