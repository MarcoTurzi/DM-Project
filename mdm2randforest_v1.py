"""
Reference:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html
https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
"""
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np


def vectorization(targetArray):
    vectorizer1 = CountVectorizer(ngram_range=(1,2))
    X1 = vectorizer1.fit_transform(targetArray)
    FeatureName_train = vectorizer1.get_feature_names_out() #array of feature names
    counts_train = X1.toarray() #array of counts of each feature

    return FeatureName_train, counts_train

#load training data
print("Loading training data...")
files_decep_train = datasets.load_files(r"C:\Users\wanne\Downloads\DM-Project-main-ass2\DM-Project-main\Part2\negative_reviws\deceptive_from_MTurk",
    categories = ["fold1", "fold2", "fold3", "fold4"], load_content = 'utf-8')
files_truth_train = datasets.load_files(r"C:\Users\wanne\Downloads\DM-Project-main-ass2\DM-Project-main\Part2\negative_reviws\truthful_from_Web",
    categories = ["fold1", "fold2", "fold3", "fold4"], load_content = 'utf-8')

#Combining training data
decep_train = files_decep_train.data #list, size 320
truth_train = files_truth_train.data #list, size 320
text_train = decep_train + truth_train #list, size 640


#Create array of class labels matching the training set
class_labels_train = np.array([])
for i in range(320):
    class_labels_train = np.append(class_labels_train,0)
for j in range(320):
    class_labels_train = np.append(class_labels_train,1)

#Amount of loops, I used 10
for rfiteration in range(10):
    print("\nIteration {}\n".format(rfiteration+1))

    #Set up Random Forest Model
    print("Training classifier...")
    RFClassifier = RF(n_estimators = 100,oob_score=False)
    vectorized = vectorization(text_train)[1]
    RFModel = RF.fit(X=vectorized, y=class_labels_train, sample_weight=None,self=RFClassifier)


    #load test data
    print("Loading test set...")
    files_decep_test = datasets.load_files(r"C:\Users\wanne\Downloads\DM-Project-main-ass2\DM-Project-main\Part2\negative_reviws\deceptive_from_MTurk",
    	categories = "fold5", load_content = 'utf-8')
    files_truth_test = datasets.load_files(r"C:\Users\wanne\Downloads\DM-Project-main-ass2\DM-Project-main\Part2\negative_reviws\truthful_from_Web",
    	categories = "fold5", load_content = 'utf-8')

    #Create array of class labels matching the test set
    class_labels_test = np.array([])
    for i in range(80):
        class_labels_test = np.append(class_labels_test,0)
    for j in range(80):
        class_labels_test = np.append(class_labels_test,1)

    #Prepare vectorization using the earlier established vocabulary
    test_data = files_decep_test.data + files_truth_test.data
    vocab = vectorization(text_train)[0]

    #Vectorization of test data, and finally test it
    testingVectorizer = CountVectorizer(vocabulary=vocab,ngram_range=(1,2))
    X2 = testingVectorizer.fit_transform(test_data)
    test_feature_count = X2.toarray()

    print("Predicting...\n")
    test_predictions = RF.predict(X=test_feature_count,self=RFClassifier)

    #Print metrics to console
    recallScore = metrics.recall_score(y_true=class_labels_test,y_pred=test_predictions)
    print("Recall: {:.4f}".format(recallScore))

    f1Score = metrics.f1_score(class_labels_test,test_predictions)
    print("F1: {:.4f}".format(f1Score))

    precisionScore = metrics.precision_score(class_labels_test,test_predictions)
    print("Precision: {:.4f}".format(precisionScore))

    genAccScore = RF.score(X=test_feature_count,y=class_labels_test,self=RFClassifier)
    print("Accuracy: {:.4f}".format(genAccScore))
