import os
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def lookup_by_val(d, val):
    inverse_dict = {v: k for k, v in d.items()}
    return inverse_dict[val]

#TUNING
train_tun_dir = ["fold1", "fold2", "fold3", "fold4"]
test_dir = ["fold5"] # I don't like it as a list but ok
train_tun_reviews = np.array([])
train_tun_labels = []
decep_reviews = np.array([])
decep_labels = []
truth_reviews = np.array([])
truth_labels = []

#dataset selection
for directory in train_tun_dir:
    for file in os.listdir("Part2\\negative_reviews\deceptive_from_MTurk\\"+directory):
        with open("Part2\\negative_reviews\deceptive_from_MTurk\\"+directory+"/"+file) as f:
            decep_reviews = np.append(decep_reviews, f.readlines())
            decep_labels.append(1)
            '''train_tun_reviews = np.append(train_tun_reviews, f.readlines())
            train_tun_labels.append(1)'''
    for file in os.listdir("Part2\\negative_reviews\\truthful_from_Web\\"+directory):
        with open("Part2\\negative_reviews\\truthful_from_Web\\"+directory+"/"+file) as f:
            '''train_tun_reviews = np.append(train_tun_reviews, f.readlines())
            train_tun_labels.append(0)'''
            truth_reviews = np.append(truth_reviews, f.readlines())
            truth_labels.append(0)
#dictionary and function to deal with conntractions
for i in range(len(decep_reviews)):
    train_tun_reviews = np.append(train_tun_reviews,decep_reviews[i])
    train_tun_labels.append(decep_labels[i])
    train_tun_reviews = np.append(train_tun_reviews,truth_reviews[i])
    train_tun_labels.append(truth_labels[i])
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

contractions_re=re.compile('(%s)' % '|'.join(contractions.keys()))

def expand_contractions(text,contractions_dict=contractions):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

#text preprocessing

for i in range(len(train_tun_reviews)):
    
    #lower case
    train_tun_reviews[i] = train_tun_reviews[i].lower()
    
    #stop words
    
    stops = stopwords.words("english")
    
    reviews_split = train_tun_reviews[i].split(" ")
    check = np.intersect1d(reviews_split, stops)

    if len(check) > 0:
        
        for x in check:
            
           train_tun_reviews[i] = train_tun_reviews[i].replace(" "+x+" "," ")
    
    #remove puntuaction
    train_tun_reviews[i] = res = re.sub(r'[^\w\s]', '', train_tun_reviews[i])
    
    #replace contractions
    train_tun_reviews[i] = expand_contractions(train_tun_reviews[i])
    
    #remove digits
    
    train_tun_reviews[i] = re.sub("\d+", "", train_tun_reviews[i])

#cross-validation split ( 10 folders: 9 train and 1 test)
folders_number = 10
folders_text = []
folders_labels = []
n_files = len(train_tun_reviews)/folders_number
'''permutations = np.random.permutation(len(train_tun_reviews))
train_tun_reviews = np.array(train_tun_reviews)
train_tun_reviews = train_tun_reviews[permutations]
train_tun_labels = np.array(train_tun_labels)
train_tun_labels = train_tun_labels[permutations]'''





#vectorizer training

vectorizer = CountVectorizer()
reviews_vectorized = vectorizer.fit_transform(train_tun_reviews)

#classifiers training and evaluation

classifiers = np.array([])
evaluations = []

#alphas
c = DecisionTreeClassifier()
path = c.cost_complexity_pruning_path(reviews_vectorized.toarray(),train_tun_labels)
ccp_a = path.ccp_alphas
print(ccp_a)

kf = KFold(n_splits=len(ccp_a) - 20)



for i in range(folders_number):
    folders_text.append(train_tun_reviews[(i)*64:(i+1)*64])
    folders_labels.append(train_tun_labels[i*64:(i+1)*64])

i = 0

train_tun_labels = np.array(train_tun_labels)

for train, test in kf.split(reviews_vectorized.toarray(), train_tun_labels):
    train_set_reviews = train_tun_reviews[train]
    train_set_labels = train_tun_labels[train]

    test_set_reviews = train_tun_reviews[test]
    test_set_labels = train_tun_labels[test]

    #text feature extraction
    train_set_reviews = vectorizer.transform(train_set_reviews).toarray()
    test_set_reviews = vectorizer.transform(test_set_reviews).toarray()
    
    #classifier training
    
    classifiers = np.append(classifiers, DecisionTreeClassifier(ccp_alpha=ccp_a[i]))
    classifiers[i].fit(train_set_reviews, train_set_labels)
    

    #prediction and evaluation
    predicted = classifiers[i].predict(test_set_reviews)
    metrics = [precision_score(test_set_labels, predicted), accuracy_score(test_set_labels, predicted), recall_score(test_set_labels, predicted), f1_score(test_set_labels, predicted)]
    
    evaluations.append(metrics)
    i = i+ 1

evaluation_means = [ np.sum(x)/len(x) for x in evaluations]

max_val = max(evaluation_means)

index = [i for i,j in enumerate(evaluation_means) if j == max_val][0]

#TEST

test_reviews = np.array([])
test_labels = []


for directory in test_dir:
    for file in os.listdir("Part2\\negative_reviews\deceptive_from_MTurk\\"+directory):
        with open("Part2\\negative_reviews\deceptive_from_MTurk\\"+directory+"/"+file) as f:
            test_reviews = np.append(test_reviews, f.readlines())
            test_labels.append(1)
    for file in os.listdir("Part2\\negative_reviews\\truthful_from_Web\\"+directory):
        with open("Part2\\negative_reviews\\truthful_from_Web\\"+directory+"/"+file) as f:
            test_reviews = np.append(test_reviews, f.readlines())
            test_labels.append(0)

for i in range(len(test_reviews)):
    
    #lower case
    test_reviews[i] = test_reviews[i].lower()
    
    #stop words
    
    stops = stopwords.words("english")
    
    reviews_split = test_reviews[i].split(" ")
    check = np.intersect1d(reviews_split, stops)
    
    if len(check) > 0:
        
        for x in check:
            
           test_reviews[i] = test_reviews[i].replace(" "+x+" "," ")
    
    #remove puntuaction
    test_reviews[i] = res = re.sub(r'[^\w\s]', '', test_reviews[i])
    
    #replace contractions
    test_reviews[i] = expand_contractions(test_reviews[i])
    
    #remove digits
    
    test_reviews[i] = re.sub("\d+", "", test_reviews[i])

test_reviews = vectorizer.transform(test_reviews).toarray()


predicted = classifiers[index].predict(test_reviews)
metrics = [precision_score(test_labels, predicted), accuracy_score(test_labels, predicted), recall_score(test_labels, predicted), f1_score(test_labels, predicted)]
print(metrics)
best_coef = {x:k for x,k in zip(classifiers[index].feature_importances_, range(len(classifiers[index].feature_importances_))) if x > 0}
best_coef = [best_coef[k] for k in list(best_coef.keys())[0:5]]
best_words = [lookup_by_val(vectorizer.vocabulary_, x) for x in best_coef]
print(best_words)
'''print("Fake")
coefs = {j:k for j,k in zip(classifiers[index].coef_[0], range(len(classifiers[index].coef_[0]))) }
coef = dict(sorted(coefs.items(), key=lambda item: item[0]))
best_coef = list(coef.keys())

print(best_coef[::-1][0:5])
best_coef =  [coef.get(k) for k in best_coef[::-1][0:5]]
print(best_coef)
words = [lookup_by_val(vectorizer.vocabulary_, k) for k in best_coef]
print(words)
print(lookup_by_val(vectorizer.vocabulary_, 100))
print("True")
coefs = {j:k for j,k in zip(classifiers[index].coef_[0], range(len(classifiers[index].coef_[0]))) }
coef = dict(sorted(coefs.items(), key=lambda item: item[0]))

best_coef = list(coef.keys())
best_coef =  [coef.get(k) for k in best_coef[0:5]]
print(best_coef)
words = [lookup_by_val(vectorizer.vocabulary_, k) for k in best_coef]
print(words)
print(lookup_by_val(vectorizer.vocabulary_, 100))'''
