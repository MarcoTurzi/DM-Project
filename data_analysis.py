import numpy as np
from assignment1_TJ import tree_grow, tree_traversals, tree_pred, tree_grow_b, tree_pred_b
"""
import timeit
start = timeit.default_timer()
"""

#load data
data_train = np.genfromtxt('eclipse-metrics-packages-2.0.csv', delimiter=';', skip_header=True)
data_test = np.genfromtxt('eclipse-metrics-packages-3.0.csv', delimiter=';', skip_header=True)

#preprocess the training data
post_release_train = data_train[:,3]
y_size = post_release_train.shape
y_train = np.zeros(y_size)
for i in range(y_size[0]):
	if post_release_train[i] == 0:
		y_train[i] = 0
	else:
		y_train[i] = 1

pre_release_train = data_train[:, 2]
pre_release_train = np.reshape(pre_release_train, [pre_release_train.shape[0], 1])
metrics_train = data_train[:, 4:44]
x_train = np.concatenate((pre_release_train, metrics_train), axis = 1)

#preprocess the testing data
post_release_test = data_test[:,3]
y_size = post_release_test.shape
y_test = np.zeros(y_size)
for i in range(y_size[0]):
	if post_release_test[i] == 0:
		y_test[i] = 0
	else:
		y_test[i] = 1

pre_release_test = data_test[:, 2]
pre_release_test = np.reshape(pre_release_test, [pre_release_test.shape[0], 1])
metrics_test = data_test[:, 4:44]
x_test = np.concatenate((pre_release_test, metrics_test), axis = 1)


#question 1
TP = 0
FP = 0
FN = 0
TN = 0
single_tree = tree_grow(x_train, y_train, 15, 5, 41)
pred_result = tree_pred(x_test, single_tree)
for i in range(y_test.shape[0]):
	if pred_result[i] == 1 and y_test[i] == 1:
		TP+=1
	elif pred_result[i] == 1 and y_test[i] == 0:
		FP+=1
	elif pred_result[i] == 0 and y_test[i] == 1:
		FN+=1
	else:
		TN+=1

precision = TP/(TP+FP)
print("The precision of question 1 is: ", precision)
recall = TP/(TP+FN)
print("The recall of question 1 is: ", recall)
accuracy = (TP + TN)/(TP+TN+FP+FN)
print("The accuracy of question 1 is: ", accuracy)


#question 2
TP = 0
FP = 0
FN = 0
TN = 0
trees = tree_grow_b(x_train, y_train, 15, 5, 41, 100)
pred_result = tree_pred_b(x_test, trees)

for i in range(y_test.shape[0]):
	if pred_result[i] == 1 and y_test[i] == 1:
		TP+=1
	elif pred_result[i] == 1 and y_test[i] == 0:
		FP+=1
	elif pred_result[i] == 0 and y_test[i] == 1:
		FN+=1
	else:
		TN+=1

precision = TP/(TP+FP)
print("The precision of question 2 is: ", precision)
recall = TP/(TP+FN)
print("The recall of question 2 is: ", recall)
accuracy = (TP + TN)/(TP+TN+FP+FN)
print("The accuracy of question 2 is: ", accuracy)

"""
stop = timeit.default_timer()
print('Time: ', stop - start)
"""

#question 3

TP = 0
FP = 0
FN = 0
TN = 0
trees = tree_grow_b(x_train, y_train, 15, 5, 6, 100)
pred_result = tree_pred_b(x_test, trees)

for i in range(y_test.shape[0]):
	if pred_result[i] == 1 and y_test[i] == 1:
		TP+=1
	elif pred_result[i] == 1 and y_test[i] == 0:
		FP+=1
	elif pred_result[i] == 0 and y_test[i] == 1:
		FN+=1
	else:
		TN+=1

precision = TP/(TP+FP)
print("The precision of question 3 is: ", precision)
recall = TP/(TP+FN)
print("The recall of question 3 is: ", recall)
accuracy = (TP + TN)/(TP+TN+FP+FN)
print("The accuracy of question 3 is: ", accuracy)



