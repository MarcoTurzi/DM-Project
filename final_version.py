"""
Tingyang Jiao   7535481
Marco Turzi 7662238
Laurian Wannee 1817531
"""

import numpy as np

"""
Node(): The basic data structure of decision tree implementation
Each Node object represents a node in decision tree
Reference: Miller, Brad and David Ranum "Problem Solving with Algorithms and Data Structures", Chapter 6.4
"""
class Node():
	def __init__(self):
		self.decision = None
		self.left = None 
		self.right = None 
		self.if_leaf = False
		self.leaf_label = 0

	#set the decision of this node; decision: a list of length 2, with the best variable and the best split of this variable
	def set_decision(self, decision):
		self.decision = decision

	#set the left child of this node; left_node: a Node object
	def set_leftNode(self, left_node):
		self.left = left_node

	#set the right child of this node; right_node: a Node object
	def set_rightNode(self, right_node):
		self.right = right_node

	#set if this node is leaf; value = True if it's leaf, False if not
	def set_leaf(self, value):
		self.if_leaf = value

	#return if this node is leaf; return Ture if it is, False if not
	def check_if_leaf(self):
		return self.if_leaf

	#return the decision of this node
	def get_decision(self):
		return self.decision

	#return the left child
	def get_leftNode(self):
		return self.left

	#return the right child
	def get_rightNode(self):
		return self.right

	#for leaf node, set leaf label; leaf_label: the predict value of this leaf
	def set_leaf_label(self, leaf_label):
		self.leaf_label = leaf_label

	#return the leaf label
	def get_leaf_label(self):
		return self.leaf_label

	

"""
tree_grow: construct a decision tree by recursion
x: 2D array, inupt data with attribute values
y: 1D array, label values
nmin: int, minimum amount of values a node must contain
minleaf: int, minimum anount of values a leaf node must contain
nfeat: amount of features, if nfeaf is less than the total amount of features in x, do random forest
Return: the root node of this decision tree
"""
def tree_grow(x, y, nmin, minleaf, nfeat):
	#create a new node
	node = Node()

	#check if x satisfies nmin, if not, set this node as leaf node and return it
	num_cases = x.shape[0]
	if num_cases < nmin:
		node.set_leaf(True)
		label = 0 if np.count_nonzero(y == 0) > np.count_nonzero(y == 1) else 1
		node.set_leaf_label(label)
		return node

	#check if all values in x belong to same label
	if np.unique(y).size == 1:
		node.set_leaf(True)
		label = 0 if np.count_nonzero(y == 0) > np.count_nonzero(y == 1) else 1
		node.set_leaf_label(label)
		return node

	#get the best split among all variables (or nfeat variables in random forest)
	best_variable = None #the index of variable with the final split
	final_split = 0
	max_quality = 0

	#get an index array of all features in x
	feature_index = np.arange(x.shape[1])
	
	#if nfeat < amount of variables, do random selection, else, do normal split
	if nfeat < x.shape[1]:
		index_nfeat = np.random.choice(feature_index, nfeat, replace = False)
	else:
		index_nfeat = feature_index

	for i in range(x.shape[1]):
		if i in index_nfeat:
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
		label = 0 if np.count_nonzero(y == 0) > np.count_nonzero(y == 1) else 1
		node.set_leaf_label(label)
		return node

	#recursively construct the tree
	left_node = tree_grow(left_values, left_lables, nmin, minleaf, nfeat)
	node.set_leftNode(left_node)
	right_node = tree_grow(right_values, right_lables, nmin, minleaf, nfeat)
	node.set_rightNode(right_node)

	return node

"""
tree_pred: use decision tree to redict the labels of input variables
x: 2D array, input variables with predictions required
tr: the root node of decision tree
Return: y, 1D array, the labels of prediction
"""
def tree_pred(x, tr):
	#create y which contains the predicted value of each variable in x
	x_coord = x.shape[0]
	y = np.zeros(x_coord)

	#do the prediction 1 by 1
	for i in range(x_coord):
		label = single_pred(x[i, :], tr)
		y[i] = label
	
	return y


"""
tree_grow_b: bagging, randomly sample from input dataset with replace, and build a decision tree on this sample,
             repeat this step to build several decision trees
x: 2D array, inupt data with attribute values
y: 1D array, label values
nmin: int, minimum amount of values a node must contain
minleaf: int, minimum anount of values a leaf node must contain
nfeat: amount of features, if nfeaf is less than the total amount of features in x, do random forest
m: the amount of trees need to be built
Return: a list of m trees
"""
def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
	trees_list = []
	for i in range(m):
		random_rows = np.random.choice(np.arange(x.shape[0]), x.shape[0], replace = True)
		x_samples = x[random_rows, :]
		y_samples = y[random_rows]
		tree = tree_grow(x_samples, y_samples, nmin, minleaf, nfeat)
		trees_list.append(tree)

	return trees_list


"""
tree_pred_b:implement tree_pred to x with each tree in the tree_list of m trees, 
             find the majority label of each variable and set it as final predicted label
x: 2D array, inupt data with attribute values
tree_list: a list of m decision trees
Return: final prediction of each variable, 1D array
"""
def tree_pred_b(x, trees_list):
	#do the prediction M times
	prediction = np.zeros([len(trees_list), x.shape[0]])
	for i in range(len(trees_list)):
		prediction[i] = tree_pred(x, trees_list[i])

	#get the majority prediction of each variable
	final_predict = np.zeros(prediction.shape[1])
	for i in range(prediction.shape[1]):
		count = np.unique(prediction[:, i], return_counts = True)
		frequency = count[1][0]/prediction[:, i].size
		if frequency >= (1 - frequency):
			final_predict[i] = count[0][0]
		else:
			final_predict[i] = count[0][1]
	return final_predict


"""
get_best_split: calculate the best split point of one feature x
x: feature values of 1 variable, array, size 1 by n
y: labels, 1D array
Return: the best split point on this feature and the quality of split
"""
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

"""
tree_traversals: print all nodes of a tree, for test only
node: the root node of decision tree
"""
def tree_traversals(node):
	print(node.get_decision())
	if node.if_leaf == False:
		tree_traversals(node.get_leftNode())
		tree_traversals(node.get_rightNode())
	else:
		return
	return

"""
single_pred: predit the label of one singel input vector, by recursion
input: features of 1 variable to be predicted
tr: root node of decision tree
Return: predicted label of this input variable
"""
def single_pred(input, tr):
	if tr.check_if_leaf():
		return tr.get_leaf_label()
	if input[tr.get_decision()[0]] < tr.get_decision()[1]:
		label = single_pred(input, tr.get_leftNode())
	else:
		label = single_pred(input, tr.get_rightNode())
	return label

