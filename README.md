# ml-movie-reviews-naive-bayes-classifier
This project is a naive bayes classifier to predict the category of moview reviews as 'Good' or 'Bad'.

- movie-reviews-analysis.py
  - main python file which has the code to train the model using the training dataset and subsequently predict on the provided testing dataset. this code uses pipelines to fit and transform the data.

- movie-pang02-train.csv
  - csv file with the required training dataset. this file is a cut down version of the file available at http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW3/

- movie-pang02-test.csv
  - csv file with the required testing dataset. this file is a cut down version of the file available at http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW3/


**SVM Kernel Importance***
- SVM is a supervised learning algorithm which can be used for both linear and non-linear datasets. For non-linear dataset SVM supports Kernels which can be help is classifying the dataset. There are
three types of Kernels:
  - Linear (used for linear dataset)
  - polynomial 
  - rbf

Typically if the number of features is high then linear kernel is a good option to use, whereas if the number of measurements is less then an rbf kernel is a good option to use.

Kernels can be thought of as functions which convert a low-dimensional data into a high-dimensional data. So in a way they make inseperable data as seperable which is very useful for non-linear datasets.