import numpy as np

class Node():
	def __init__(self):
		self.decision = None
		self.left = None
		self.right = None
		self.if_leaf = False
		self.leaf_label = 0

	def set_decision(self, decision):
		self.decision = decision

	def set_leftNode(self, left_node):
		self.left = left_node

	def set_rightNode(self, right_node):
		self.right = right_node

	def set_leaf(self, value):
		self.if_leaf = value

	def check_if_leaf(self):
		return self.if_leaf

	def get_decision(self):
		return self.decision

	def get_leftNode(self):
		return self.left

	def get_rightNode(self):
		return self.right

	def get_leaf_label(self):
		return self.leaf_label

	def set_leaf_label(self, leaf_label):
		self.leaf_label = leaf_label


	

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

def tree_pred(x, tr):
	#create y which is the predicted value of each item in x
	x_coord = x.shape[0]
	y = np.zeros(x_coord)

	for i in range(x_coord):
		label = single_pred(x[i, :], tr)
		y[i] = label
	
	return y

#predit the label of one singel input vector
def single_pred(input, tr):
	if tr.check_if_leaf():
		return tr.get_leaf_label()
	if input[tr.get_decision()[0]] < tr.get_decision()[1]:
		label = single_pred(input, tr.get_leftNode())
	else:
		label = single_pred(input, tr.get_rightNode())
	return label

def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
	trees_list = []
	for i in range(m):
		random_rows = np.random.choice(np.arange(x.shape[0]), x.shape[0], replace = True)
		x_samples = x[random_rows, :]
		y_samples = y[random_rows]
		tree = tree_grow(x_samples, y_samples, nmin, minleaf, nfeat)
		trees_list.append(tree)

	return trees_list

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





#test cases

credit_data = np.genfromtxt('pima.txt', delimiter=',', skip_header=True)
size = credit_data.shape
x = credit_data[:, 0:(size[1] - 1)]
y = credit_data[:, -1]


samples = tree_grow_b(x, y, 20, 5, 8, 5)
result = tree_pred_b(x, samples)
print(np.unique(result, return_counts = True))

tree = tree_grow(x, y, 20, 5, 8)
result = tree_pred(x, tree)
print(np.unique(result, return_counts = True))
print(np.count_nonzero(y == 0))
print(np.count_nonzero(y == 1))

