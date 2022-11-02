import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif


#load training data
files_decep_train = datasets.load_files("C:/Users/Tingyang/Documents/Utrecht University/Courses/Data Mining/Assignment 2/op_spam_v1.4/negative_polarity/deceptive_from_MTurk",
	categories = ["fold1", "fold2", "fold3", "fold4"], load_content = 'utf-8')
files_truth_train = datasets.load_files("C:/Users/Tingyang/Documents/Utrecht University/Courses/Data Mining/Assignment 2/op_spam_v1.4/negative_polarity/truthful_from_Web",
	categories = ["fold1", "fold2", "fold3", "fold4"], load_content = 'utf-8')

#raw training data
decep_train = files_decep_train.data #list, size 320
truth_train = files_truth_train.data #list, size 320
text_train = decep_train + truth_train #list, size 640

#creating the labels array
decep_labels = np.zeros(len(decep_train))
truth_labels = np.ones(len(truth_train))
train_labels = np.concatenate((decep_labels, truth_labels))

#best 5 decep features
vectorizer1 = CountVectorizer()
X1 = vectorizer1.fit_transform(decep_train)
FeatureName_train = vectorizer1.get_feature_names_out() #array of feature names
counts_train = X1.toarray() #array of counts of each feature


#calculate mutural information
mutual_info = mutual_info_classif(counts_train, decep_labels)
mutual_info = np.resize(mutual_info, (1, mutual_info.shape[0]))
total_train = np.concatenate((counts_train, mutual_info), axis = 0)


#sort the array by mutural information
index_sorted = np.argsort(total_train[-1, :])
total_train_sorted = total_train[:, index_sorted]
FeatureName_sorted = FeatureName_train[index_sorted]

FeatureName_selected = FeatureName_sorted[-5:]

print(FeatureName_sorted == FeatureName_train)

print(FeatureName_selected)

"""
#best 5 truth features
vectorizer1 = CountVectorizer()
X1 = vectorizer1.fit_transform(truth_train)
FeatureName_train = vectorizer1.get_feature_names_out() #array of feature names
counts_train = X1.toarray() #array of counts of each feature


#calculate mutural information
mutual_info = mutual_info_classif(counts_train, truth_labels)
mutual_info = np.resize(mutual_info, (1, mutual_info.shape[0]))
total_train = np.concatenate((counts_train, mutual_info), axis = 0)


#sort the array by mutural information
index_sorted = np.argsort(total_train[-1, :])
total_train_sorted = total_train[:, index_sorted]
FeatureName_sorted = FeatureName_train[index_sorted]

FeatureName_selected = FeatureName_sorted[-5:]

print(FeatureName_selected)
"""



