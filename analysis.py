import numpy as np
import pandas as pd
from requests import post 
from sklearn.model_selection import train_test_split
from marco import tree_grow_b,tree_pred, Node, tree_pred_b
from treelib import Tree,Node


def get_labels(data):

    post_column = data["post"].to_numpy()
    return [0 if i == 0 else 1 for i in post_column ]

def extract_obs(data):

    obs = data.to_numpy()
    obs = np.concatenate(( np.array([obs[:,2]]).T, obs[:,4:]), axis=1)
    return obs

def evaluate_confusion_matrix(y, y_pred):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for yy, yp in zip(y, y_pred):
        if yy == 0 and yp == 0:
            tp = tp + 1
        if yy == 0 and yp == 1:
            fn = fn + 1
        if yy == 1 and yp == 0:
            fp = fp + 1
        if yy == 1 and yp == 1:
            tn = tn + 1
    return np.append([[tp, fn]], [[fp, tn]], axis = 0)

def evaluate_precision(y , y_pred):

    tp = 0
    fp = 0
    for yy, yp in zip(y, y_pred):
        if yy==0 and yp == 0:
            tp = tp + 1
        if yy==1 and yp == 0:
            fp = fp + 1
    
    return tp/(tp + fp)

def evaluate_recall(y, y_pred):

    tp = 0
    fn = 0

    for yy, yp in zip(y, y_pred):
        if yy== 0 and yp == 0:
            tp = tp + 1
        if yy==0 and yp == 1:
            fn = fn + 1
    
    return tp/(tp + fn)

def evaluate_accuracy(y, y_pred):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for yy, yp in zip(y, y_pred):
        if yy == 0 and yp == 0:
            tp = tp + 1
        if yy == 0 and yp == 1:
            fn = fn + 1
        if yy == 1 and yp == 0:
            fp = fp + 1
        if yy == 1 and yp == 1:
            tn = tn + 1
    
    return (tp+ tn)/(tp+tn+fn+fp)


def get_data():

    data_train = pd.read_csv("eclipse-metrics-packages-2.0.csv", header=0, sep=";")
    labels_train = get_labels(data_train)
    observations_train = extract_obs(data_train)
    data_test = pd.read_csv("eclipse-metrics-packages-3.0.csv", header=0, sep=";")
    labels_test = get_labels(data_test)
    observations_test = extract_obs(data_test)
    return observations_test, observations_test, labels_train, labels_test

if __name__ == "__main__":

    #extract data
    x_train, x_test, y_train, y_test = get_data()
    print(sum(y_train)/len(y_train))

    #train classification tree
    trees = tree_grow_b(x_train,y_train, 15,5,41,1)
    print(trees)
    
    #show tree
    for i in trees:
        tree = Tree()
        i.graph(tree, "root")
        tree.show()
    
    y_pred = tree_pred(x_test, trees[0])
    

    print("Precision : ", round(evaluate_precision(y_test, y_pred),3))
    print("Recall: ", round(evaluate_recall(y_test, y_pred), 3) )
    print("Accuracy: ",round( evaluate_accuracy(y_test, y_pred), 3))
    print("Confusion matrix :", evaluate_confusion_matrix(y_test, y_pred))

    trees = tree_grow_b(x_train,y_train, 15,5,41,100)
    y_pred = tree_pred_b(x_test, trees)

    print("Precision : ", round(evaluate_precision(y_test, y_pred),3))
    print("Recall: ", round(evaluate_recall(y_test, y_pred), 3) )
    print("Accuracy: ",round( evaluate_accuracy(y_test, y_pred), 3))
    print("Confusion matrix :", evaluate_confusion_matrix(y_test, y_pred))

    trees = tree_grow_b(x_train,y_train, 15,5,6,100)
    y_pred = tree_pred_b(x_test, trees)

    print("Precision : ", round(evaluate_precision(y_test, y_pred),3))
    print("Recall: ", round(evaluate_recall(y_test, y_pred), 3) )
    print("Accuracy: ",round( evaluate_accuracy(y_test, y_pred), 3))
    print("Confusion matrix :", evaluate_confusion_matrix(y_test, y_pred))

