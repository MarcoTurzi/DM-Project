import numpy as np
import numpy.random as npr
from treelib import Tree
import random

class Node():
    def __init__(self):
        self.decision = None
        self.left = None
        self.right = None
        self.if_leaf = False


    def graph(self, tree, pre_dec):
        if pre_dec == "root":
            ran = str(random.randint(1,100))
            tree.create_node(str(self.decision),str(self.decision)+ran)
            self.left.graph(tree, str(self.decision)+ran)
            self.right.graph(tree, str(self.decision)+ran)
        else:
            if self.if_leaf == False:
                ran = str(random.randint(1,100))
                tree.create_node(str(self.decision),str(self.decision)+ran, parent=pre_dec)
                self.left.graph(tree, str(self.decision)+ran)
                self.right.graph(tree, str(self.decision)+ran)
            else:
                ran = str(random.randint(1,100))
                tree.create_node("leaf"+pre_dec,"leaf"+pre_dec+ran, parent=pre_dec)

    def set_decision(self, decision):
        self.decision = decision

    def set_leftNode(self, left_node):
        self.left = left_node

    def set_rightNode(self, right_node):
        self.right = right_node

    def set_leaf(self, value):
        self.if_leaf = value

    def get_decision(self):
        return self.decision

    def get_leftNode(self):
        return self.left

    def get_rightNode(self):
        return self.right

# return the features to be checked by bestsplit 
def select_features(x, nfeat):
    #if nfeat value is too high it becames equal to the number of feature of the subset
    nfeat = nfeat if x.shape[1] >= nfeat else x.shape[1]
    #array containing the features
    n_col = [i for i in range(x.shape[1])]
    real_col = []
    #pick the nfeat features randomly
    for i in range(nfeat):
        real_col.append(n_col.pop(random.randint(0,len(n_col)- 1)))
    return np.array(real_col)


def tree_grow(x, y, nmin, minleaf, nfeat):
    #create a new node
    node = Node()

    #check if x satisfies nmin, if not, set this node as leaf node and return it
    num_cases = x.shape[0]
    if num_cases < nmin:
        node.set_leaf(True)
        return node

    #check if all values in x belong to same label (for credit.txt only, not for assignment part 2)
    if np.unique(y).size == 1:
        node.set_leaf(True)
        return node

    #get the best split among nfeat variables
    best_variable = None #the index of variable with the final split
    final_split = 0
    max_quality = 0
    
    #set of features to be checked
    x_nfeat = select_features(x,nfeat)
    
    for i in x_nfeat:
        best_split, split_quality = get_best_split(x[:, i], y)
        if split_quality > max_quality:
            max_quality = split_quality
            final_split = best_split
            best_variable = i

    node.set_decision([best_variable, final_split])

    #operate the split
    #left child node
    left_values = x[x[:, best_variable] < final_split]
    left_lables = y[x[:, best_variable] < final_split]

    #right child node
    right_values = x[x[:, best_variable] > final_split]
    right_lables = y[x[:, best_variable] > final_split]

    #check if this split satisfies minleaf
    if (left_lables.size < minleaf) or (right_lables.size < minleaf):
        node.set_leaf(True)
        return node

    #recursively construct the tree
    left_node = tree_grow(left_values, left_lables, nmin, minleaf, nfeat)
    node.set_leftNode(left_node)
    right_node = tree_grow(right_values, right_lables, nmin, minleaf, nfeat)
    node.set_rightNode(right_node)

    return node

def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    '''
    description:  tree_grow_b generates m decision trees from the dataset [x,y] 
    
    input:        feature set, label set, number of observations to allow a split, minimum number of observation for a leaf node, number of column to consider for every split, number of subsets
    output:       set of trees
    '''
    #shuffle the dataset
    t_set = np.array([np.append(xx, yy) for xx,yy in zip(x,y)])
    npr.shuffle(t_set)
    #creates m subset from the dataset
    bootstraps = np.array(np.array_split(t_set, m)) 
    #for every subset generates a tree
    trees = np.array([])
    for bs in bootstraps:
        trees = np.append(trees, tree_grow(bs[:,:bs.shape[1] -1], bs[:,bs.shape[1] -1], nmin,minleaf, nfeat))
    return trees
    
    
    
#calculate the best split point of one feature x
#x: feature values of 1 variable, size 1 by n
#y: labels
def get_best_split(x, y):
    #calculate all candidate split points
    x_sorted = np.sort(np.unique(x))
    size_Xsorted = x_sorted.size
    x_splitpoints = (x_sorted[0:(size_Xsorted-1)] + x_sorted[1:size_Xsorted])/2

    #calculate the original gini index before spliting
    count = np.unique(y, return_counts = True)
    frequency = count[1][0]/y.size
    gini_original = frequency * (1 - frequency)

    best_split = 0
    max_quality = 0

    #for each split_point, calculate its quality, update the max_quality
    for split_point in x_splitpoints:
        #get the gini_index of the left subset
        left_subset = y[x < split_point]
        left_count = np.unique(left_subset, return_counts = True)
        left_frequency = left_count[1][0]/left_subset.size
        gini_left = left_frequency * (1 - left_frequency)

        #get the gini_index of the right subset
        right_subset = y[x > split_point]
        right_count = np.unique(right_subset, return_counts = True)
        right_frequency = right_count[1][0]/right_subset.size
        gini_right = right_frequency * (1 - right_frequency)

        split_quality = gini_original - (left_subset.size/y.size)*gini_left - (right_subset.size/y.size)*gini_right

        if split_quality > max_quality:
            max_quality = split_quality
            best_split = split_point

    return best_split, max_quality



#print all nodes of a tree, for test only
def tree_traversals(node):
    print(node.get_decision())
    if node.if_leaf == False:
        tree_traversals(node.get_leftNode())
        tree_traversals(node.get_rightNode())
    else:
        return
    return

#test cases

credit_data = np.genfromtxt('pima.txt', delimiter=',', skip_header=True)
size = credit_data.shape
x = credit_data[:, 0:(size[1] - 1)]
y = credit_data[:, -1]

root_nodes = tree_grow_b(x, y, 12, 8, 9,3)
for tree_node in root_nodes:
    tree = Tree()
    tree_node.graph(tree, "root")
    tree.show()

tree_traversals(root_node)
