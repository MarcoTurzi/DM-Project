import numpy as np
import pandas as pd

def tree_grow(x,y,nmin,minleaf,nfeat):
    nodes = np.array([])
    np.append(nodes, x.join(y))
    print(x.join(y))

    #Framework so that early stopping can be implemented in the future
    proceedCheck = True
    while proceedCheck==True:
        getBestSplit(x,y)
        break #for now, as early stopping is not active yet




def getBestSplit(x,y):
    for i in range( x.shape[1] ):
        print("Evaluating {}".format(x.columns[i]))
        values = np.array([x.loc[:,x.columns[i]]])
        values = np.sort(values)

        for j in len(values):
            computeSplit(values[j]); #Pseudocode
            #Duplicates not considered
            #How to store the results?
            #Review the gini score and how it works





dataTable = pd.read_table(r'C:\Users\wanne\Downloads\credit.txt',delimiter=",")

tree_grow(dataTable.loc[:,"age":"gender"],dataTable.loc[:,"class"], 2, 1, len(dataTable))
