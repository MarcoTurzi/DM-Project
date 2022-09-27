import numpy as np

class Node():
	def __init__(self):
		self.decision = None
		self.left = None
		self.right = None
		self.if_leaf = False

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

	#get the best split among all variables
	best_variable = None #the index of variable with the final split
	final_split = 0
	max_quality = 0

	for i in range(x.shape[1]):
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

credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)
size = credit_data.shape
x = credit_data[:, 0:(size[1] - 1)]
y = credit_data[:, -1]

root_node = tree_grow(x, y, 0, 0, 0)
tree_traversals(root_node)








