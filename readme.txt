Part A:
The classifier that is trained and predicts whether a message is SPAM or HAM based on a number of features, 
uses a Naive Bayes implementation over K-cluster in order to maximise the accuracy of the predictor.

MyClassifier has a main train and predict functions, and two functions: estimate_log_priors and estimate_log_class_conditional_likelihoods that are helper function for train.

estimate_log_priors calculates the logarithm of the empirical class priors based on the train data labels.
estimate_log_class_conditional_likelihoods calculates the logarithm of the empirical class-conditional likelihoods used in Naives Bayes based on conditional probability. This function uses the list of features and the binary response variables in order to calculate theta, which contains the probability of feature i appearing in a sample given that its in a certain class. I used Laplace smoothing with an alpha value of 1 in order to avoid zero probabilities.
We also take the logarithm of the posterior distributors to increase the numerical stability of the algorithm, this does not change the position of the maximum of the input function. That is why taking the logarithm of a product is like summing their logarithms.

The train function used the train data from the csv file to return both the log_class_priors and the log_class_conditionals. These are stored as attributes of the class in order to be accessible by the predict function.

The predict function takes in the test data from the test csv file. It then uses the log_class_priors and the log_class_conditionals calculates with the train data to predict the response based on the feature. It calculates the probability of each sample being ham or spam given the features it has.

I decided to call the train function right after creating the classifier object but an alternative could have been calling it from the constructor which would require passing in the train data into the constructor.


Part B:
When coding the BetterFeatureExtractor, I created another script to test it since MyClassifier did not support multinomial classes (0-9 digits) as it was designed for SPAM or HAM (0 or 1). I used the LogisticRegression from sklearn.linear_model, which gave a Basic accuracy of 0.8.
I attempted to implement certain features such as symmetrical, black and white pixels and some more complex features such as upper or lower loop (9 is a lower loop, 6 is an upper loop). However these did not seem to change the accuracy values that I was getting.
I initially thought to convert these features into binary, but even after those changes my accuracy difference was still 0.
In this version of the notebook I am leaving the average, white, black pixels and symmetrical feature just in case the problem was with my classifier.



