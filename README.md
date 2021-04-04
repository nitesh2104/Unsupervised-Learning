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
- [ ] PCA 
- [ ] ICA
- [ ] Randomized Projections
- [ ] Lasso Selection OR Feature Aglomeration


#### Dataset
- [ ] Phone Price Prediction
- [ ] Salary Prediction

--- 
#### STEPS and Guidelines
-    [x] Run the clustering algorithms on the datasets and describe what you see.
-    [ ] Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
-    [ ] Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it. Yes, that’s 16 combinations of datasets, dimensionality reduction, and clustering method. You should look at all of them, but focus on the more interesting findings in your report.
-    [ ] Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 (if you've reused the datasets from assignment #1 to do experiments 1-3 above then you've already done this) and rerun your neural network learner on the newly projected data.
-    [ ] Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction algorithms (you've probably already done this), treating the clusters as if they were new features. In other words, treat the clustering algorithms as if they were dimensionality reduction algorithms. Again, rerun your neural network learner on the newly projected data.

#### Requirements
-    [ ] a discussion of your datasets, and why they're interesting: If you're using the same datasets as before at least briefly remind us of what they are so we don't have to revisit your old assignment write-up... and if you aren't well that's a whole lot of work you're going to have to recreate from assignment 1 isn't it?
-    [ ] explanations of your methods: for example, how did you choose k?
-    [ ] a description of the kind of clusters that you got.
-    [ ] analyses of your results. Why did you get the clusters you did? Do they make "sense"? If you used data that already had labels (for example data from a classification problem from assignment #1) did the clusters line up with the labels? Do they otherwise line up naturally? Why or why not? Compare and contrast the different algorithms. What sort of changes might you make to each of those algorithms to improve performance? How much performance was due to the problems you chose? Be creative and think of as many questions you can, and as many answers as you can. Take care to justify your analysis with data explicitly.
-    [ ] Can you describe how the data look in the new spaces you created with the various algorithms? For PCA, what is the distribution of eigenvalues? For ICA, how kurtotic are the distributions? Do the projection axes for ICA seem to capture anything "meaningful"? Assuming you only generate k projections (i.e., you do dimensionality reduction), how well is the data reconstructed by the randomized projections? PCA? How much variation did you get when you re-ran your RP several times (I know I don't have to mention that you might want to run RP many times to see what happens, but I hope you forgive me)?
-    [ ] When you reproduced your clustering experiments on the datasets projected onto the new spaces created by ICA, PCA, and RP, did you get the same clusters as before? Different clusters? Why? Why not?
-    [ ] When you re-ran your neural network algorithms were there any differences in performance? Speed? Anything at all?

