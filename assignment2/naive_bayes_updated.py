"""
Reference:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html
https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn-feature-selection-mutual-info-classif
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score
"""
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score


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

#create candidate of sparse removing frenquency
frequency_diff = 0.01
frequency_amount = 10
frequency_candidates = []
for i in range(frequency_amount):
	frequency_candidates.append(i*frequency_diff)


def select_HyperParameter(text_train, train_labels, frequency_candidates):
	best_frequency = None
	best_selection = None
	best_score = 0
	best_x = None
	best_featureName = None

	for frequency in frequency_candidates:
		#pre-process training data
		vectorizer1 = CountVectorizer(min_df = frequency)
		X1 = vectorizer1.fit_transform(text_train)
		FeatureName_train = vectorizer1.get_feature_names_out() #array of feature names
		counts_train = X1.toarray() #array of counts of each feature

		#calculate mutural information
		mutual_info = mutual_info_classif(counts_train, train_labels)
		mutual_info = np.resize(mutual_info, (1, mutual_info.shape[0]))
		total_train = np.concatenate((counts_train, mutual_info), axis = 0)

		#sort the array by mutural information
		index_sorted = np.argsort(total_train[-1, :])
		total_train_sorted = total_train[:, index_sorted]
		FeatureName_sorted = FeatureName_train[index_sorted]

		#calculate the candidate of feature selections
		selection_diff = 50
		selection_amount = 5
		selection_candidates = []
		for i in range(selection_amount):
			selection_candidates.append(i*selection_diff)

		for abandon_features in selection_candidates:
			remain_features = counts_train.shape[1] - abandon_features
			total_train_selected = total_train_sorted[:, -remain_features:]
			FeatureName_selected = FeatureName_sorted[-remain_features:]

			#calculate cross validation score
			target_x = total_train_selected[0:counts_train.shape[0], :]
			classifier = MultinomialNB()
			cv_scores = cross_val_score(estimator = classifier, X = total_train_selected[0:counts_train.shape[0], :], 
				y = train_labels, cv = 10)
		
			cv_scores_avg = np.average(cv_scores)
			
			if cv_scores_avg > best_score:
				best_frequency = frequency
				best_selection = remain_features
				best_score = cv_scores_avg
				best_x = target_x
				best_featureName = FeatureName_selected
	print("The result of hyper-parameter seletion is removing features with documents frequency less than %f, then select best %d features with mutural information" 
		%(best_frequency, best_selection))
	return best_x, best_featureName

best_x, best_featureName= select_HyperParameter(text_train, train_labels, frequency_candidates)
print(best_x.shape)
print(best_featureName.shape)



#training naive bayes classifier
classifier = MultinomialNB()
classifier.fit(best_x, train_labels)

#load testing data
files_decep_test = datasets.load_files("C:/Users/Tingyang/Documents/Utrecht University/Courses/Data Mining/Assignment 2/op_spam_v1.4/negative_polarity/deceptive_from_MTurk",
	categories = ["fold5"], load_content = 'utf-8')
files_truth_test = datasets.load_files("C:/Users/Tingyang/Documents/Utrecht University/Courses/Data Mining/Assignment 2/op_spam_v1.4/negative_polarity/truthful_from_Web",
	categories = ["fold5"], load_content = 'utf-8')


#raw testing data
decep_test = files_decep_test.data #list, size 80
truth_test = files_truth_test.data #list, size 80
#text_train = decep_train + truth_train


#pre-process training data
vectorizer2 = CountVectorizer(vocabulary = best_featureName)
X2 = vectorizer2.fit_transform(decep_test)
counts_decep_test = X2.toarray() #array of counts of each feature

vectorizer3 = CountVectorizer(vocabulary = best_featureName)
X3 = vectorizer3.fit_transform(truth_test)
counts_truth_test = X3.toarray() #array of counts of each feature


#perdict on testing set
testResult_decep = classifier.predict(counts_decep_test)
countsResult_decep = np.unique(testResult_decep, return_counts = True)
print(countsResult_decep)

testResult_truth = classifier.predict(counts_truth_test)
countsResult_truth = np.unique(testResult_truth, return_counts = True)
print(countsResult_truth)

precision = (countsResult_truth[1][1] + countsResult_decep[1][0])/160
print(precision)


