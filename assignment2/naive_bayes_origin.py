"""
Reference:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html
https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
"""
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


#load training data
files_decep_train = datasets.load_files("C:/Users/Tingyang/Documents/Utrecht University/Courses/Data Mining/Assignment 2/op_spam_v1.4/negative_polarity/deceptive_from_MTurk",
	categories = ["fold1", "fold2", "fold3", "fold4"], load_content = 'utf-8')
files_truth_train = datasets.load_files("C:/Users/Tingyang/Documents/Utrecht University/Courses/Data Mining/Assignment 2/op_spam_v1.4/negative_polarity/truthful_from_Web",
	categories = ["fold1", "fold2", "fold3", "fold4"], load_content = 'utf-8')

#raw training data
decep_train = files_decep_train.data #list, size 320
truth_train = files_truth_train.data #list, size 320
text_train = decep_train + truth_train #list, size 640

#pre-process training data
vectorizer1 = CountVectorizer()
X1 = vectorizer1.fit_transform(text_train)
FeatureName_train = vectorizer1.get_feature_names_out() #array of feature names
counts_train = X1.toarray() #array of counts of each feature

#creating the label array
decep_labels = np.zeros(len(decep_train))
truth_labels = np.ones(len(truth_train))
train_labels = np.concatenate((decep_labels, truth_labels))

#training naive bayes classifier
classifier = MultinomialNB()
classifier.fit(counts_train, train_labels)


#load testing data
files_decep_test = datasets.load_files("C:/Users/Tingyang/Documents/Utrecht University/Courses/Data Mining/Assignment 2/op_spam_v1.4/negative_polarity/deceptive_from_MTurk",
	categories = ["fold5"], load_content = 'utf-8')
files_truth_test = datasets.load_files("C:/Users/Tingyang/Documents/Utrecht University/Courses/Data Mining/Assignment 2/op_spam_v1.4/negative_polarity/truthful_from_Web",
	categories = ["fold5"], load_content = 'utf-8')


#raw testing data
decep_test = files_decep_test.data #list, size 80
truth_test = files_truth_test.data #list, size 80
#text_train = decep_train + truth_train #list, size 640


#pre-process training data
vectorizer2 = CountVectorizer(vocabulary = FeatureName_train)
X2 = vectorizer2.fit_transform(decep_test)
counts_decep_test = X2.toarray() #array of counts of each feature

vectorizer3 = CountVectorizer(vocabulary = FeatureName_train)
X3 = vectorizer3.fit_transform(truth_test)
counts_truth_test = X3.toarray() #array of counts of each feature


#perdict on testing set
testResult_decep = classifier.predict(counts_decep_test)
countsResult_decep = np.unique(testResult_decep, return_counts = True)
print(countsResult_decep)

testResult_truth = classifier.predict(counts_truth_test)
countsResult_truth = np.unique(testResult_truth, return_counts = True)
print(countsResult_truth)

precision = countsResult_truth[1][1]/(countsResult_truth[1][1] + countsResult_truth[1][0])
print(precision)





"""
corpus = ['This is the first document.', 'This document is the second document.', 
'And this is the third one.','Is this the first document?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print(X.toarray())
"""
