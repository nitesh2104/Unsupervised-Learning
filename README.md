### __Unsupervised Learning and Dimensionality Reduction__


##### In this project, we explore unsupervised learning algorithms and perform Dimensionality Reduction to obtain a subset of features with most information

--- 
#### *---How to Run---*
- Each requirement is contained within the respective directory (KMeans, PCA, ICA etc.)
- Run this cmd from the root directory: ```jupyter-lab```
- After the browser opens - open the file and run all cells
- Note: np.random.seed(0) is already added to ensure output consistent runs
---

#### Algorithms
- [x] k-means clustering
- [x] Expectation Maximization
- [x] PCA 
- [x] ICA
- [x] Randomized Projections
- [x] IPCA


#### Dataset
- [x] Phone Price Prediction
- [x] Salary Prediction

--- 
#### STEPS and Guidelines
-    [x] Run the clustering algorithms on the datasets 
-    [x] Apply the dimensionality reduction algorithms to the two datasets
-    [x] Reproducing the clustering experiments
-    [x] Applying the dimensionality reduction algorithms and reruning the neural network learner on the newly projected data.
-    [x] Applying the clustering algorithms to the same dataset to which we just applied the dimensionality reduction algorithms, treating the clusters as if they were new features. In other words, treat the clustering algorithms as if they were dimensionality reduction algorithms. Again, rerun the neural network learner on the newly projected data.

#### Requirements
-    [x] a discussion of the datasets, and why they're interesting: If we're using the same datasets as before at least briefly remind us of what they are so we don't have to revisit the old assignment write-up... and if we aren't well that's a whole lot of work we're going to have to recreate from assignment 1 isn't it?
-    [x] explanations of the methods: for example, how did we choose k?
-    [x] a description of the kind of clusters that we got.
-    [x] analyses of the results
-    [x] Describe how the data looks in the new spaces which is created with the various algorithms? For PCA, what is the distribution of eigenvalues? For ICA, how kurtotic are the distributions? Do the projection axes for ICA seem to capture anything "meaningful"? Assuming we only generate k projections (i.e., we do dimensionality reduction), how well is the data reconstructed by the randomized projections? PCA? How much variation did we get when we re-ran the RP several times (I know I don't have to mention that we might want to run RP many times to see what happens, but I hope we forgive me)?
-    [x] When the data reproduces the clustering experiments on the datasets projected onto the new spaces created by ICA, PCA, and RP, the clusters same as before ? Different clusters? Why? Why not?
-    [x] When we re-ran the neural network algorithms were there any differences in performance? Speed? Anything at all?
