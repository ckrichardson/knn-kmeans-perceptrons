import numpy as np
import matplotlib.pyplot as plt

def perceptron_train(X, Y):
	weights = np.array([0.0, 0.0])
	bias = np.array([0])
	updated = True
	index = 0


	while updated:
		updated = not updated

		for X_train, Y_train in zip(X, Y):
			# print(weights)
			# print(str(bias) + "\n")
			a = calculate_activation(weights, X_train, bias)
			update = calculate_update(a, Y_train)
			if update:
				for x in range(len(weights)):
					weights[x] = weights[x] + X_train[x] * Y_train[0]

				bias[0] = bias[0] + Y_train[0]
				updated = True

			# print("X:   " , tuple(X_train) , "Y:   " , tuple(Y_train) , "A:   " , str(a) , "WEIGHTS:   " , str(weights[0]) , str(weights[1]) , "   BIAS:   " , str(bias))

	# print(weights)
	# print(bias)
	return (weights, bias)


def perceptron_test(X_test, Y_test, w, b):
	total = len(X_test)
	correct = 0
	prediction = 0

	for x in range(len(X_test)):
		a = calculate_activation(w, X_test[x], b)
		if a <= 0:
			prediction = -1
		else:
			prediction = 1

		# print(prediction)

		if prediction == Y_test[x]:
			correct += 1

	return correct / total

def calculate_activation(weights, X, bias):
	activation = 0

	for x in range(len(weights)):
		activation += weights[x] * X[x] 

	activation += bias[0]

	return activation

def calculate_update(a, Y_train_current):
	if a * Y_train_current <= 0:
		return True 

	return False

if __name__ == "__main__":
	X = np.array([[0, 1], [1, 0], [5, 4], [1, 1], [3, 3], [2, 4], [1, 6]])
	Y = np.array([[1], [1], [-1], [1], [-1], [-1], [-1]])
	X_2 = np.array([[-2, 1], [1, 1], [1.5, -0.5], [-2, -1], [-1, -1.5], [2, -2]])
	Y_2 = np.array([[1], [1], [1], [-1], [-1], [-1]])
	W = perceptron_train(X, Y)
	W_2 = perceptron_train(X_2, Y_2)
	print(W)
	print(W_2)
	test_accuracy = perceptron_test(X, Y, W[0], W[1])
	test_accuracy_2 = perceptron_test(X_2, Y_2, W_2[0], W_2[1])
	# print(W)
	# print(test_accuracy)
	# print(W_2)
	# print(test_accuracy_2)
	# 


	"For dataset 1"
	# x_1 = [0,1,1]
	# x_2 = [5,3,2,1]

	# y_1 = [1,0,1]
	# y_2 = [4,3,4,6]

	# x = np.linspace(-5,5,2)	
	# y = (-(W[1]/W[0][1]) / (W[1]/W[0][0])) * x + (-W[1] / W[0][1])

	# plt.plot(x, y, '-r', label="Decision Boundary")
	# plt.scatter(x_1, y_1, color="green")
	# plt.scatter(x_2, y_2, color="blue")

	# plt.show()

	x_1 = [-2,1,1.5]
	x_2 = [-2,-1,2]

	y_1 = [1,1,-0.5]
	y_2 = [-1,-1.5,-2]

	x = np.linspace(-5,5,2)	
	y = (-(W_2[1]/W_2[0][1]) / (W_2[1]/W_2[0][0])) * x + (-W_2[1] / W_2[0][1])

	plt.plot(x, y, '-r', label="Decision Boundary")
	plt.scatter(x_1, y_1, color="green")
	plt.scatter(x_2, y_2, color="blue")

	plt.show()