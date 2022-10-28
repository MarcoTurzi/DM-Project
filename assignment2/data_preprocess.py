"""
Reference:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html
https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
"""
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer


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


