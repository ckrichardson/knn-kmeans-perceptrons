import numpy as np
from random import choice
from math import sqrt

import matplotlib.pyplot as plt

def K_Means(X, K):
	X = X.astype(float)
	picked = False
	centers = np.zeros(shape=(K, len(X[0])), dtype=float)
	picked_index = 0
	cluster_points_prior = None
	changing = True
	clusters = np.zeros(shape=(len(X), len(X[0])), dtype=float)

	while picked_index < K:
		sample = choice(X)
		if sample not in centers:
			centers[picked_index] = sample
			picked_index += 1

	while changing:
		clusters = calculate_clusters(K, X, centers, clusters)
		centers, changing = recalculate_centers(X, centers, clusters)

	return centers

def calculate_clusters(K, X, centers, clusters):
	
	distances = np.zeros(shape=(len(centers), 1), dtype=float)
	cluster_points = np.zeros(shape=(len(X), len(X[0])), dtype=float)


	for x in range(len(X)):
		nearest = 0
		for y in range(len(distances)):
			distances[y] = calculate_distance(centers[y], X[x])

		cluster_point = centers[np.argmin(distances)]
		cluster_points[x] = cluster_point

	# print(cluster_points)
	# print(clusters)

	return cluster_points

def recalculate_centers(X, centers, clusters):
	# centers = np.zeros(shape=(K, len(X[0])), dtype=float)
	total= 0
	average = 0
	num_in_cluster = 0
	temp_centers = centers

	# print("BEFORE", centers)

	for y in range(len(centers)):
		total = 0
		average = 0
		num_in_cluster = 0
		for x, classification in zip(X, clusters):
			# print(x, classification, center, classification==center)
			if np.array_equal(classification, centers[y]):
				total += x
				num_in_cluster += 1

		average = total / num_in_cluster
		centers[y] = average
		# print(average)

	# print("AFTER", centers)

	if np.array_equal(temp_centers, centers):
		return centers, False

	return centers, True

			 
def calculate_distance(center, point):
	total = 0
	for x in range(len(center)):
		total += (point[x] - center[x]) ** 2

	return sqrt(total)

def K_Means_better(X, K):
	temp = []
	list_of_results = np.zeros(shape=(K, len(X[0])), dtype=float)
	index = 0
	total = 0
	total_max = 0

	for x in range(1000):
		C = K_Means(X, K)
		temp.append(C)
		# list_of_results = np.append(list_of_results, C)
		# print(list_of_results)

	real = np.asarray(temp)

	temp=None

	for element in real:
		index = 0
		total = 0

		for check in real:
			if (element==check).all():
				total += 1
				# print(element, check)

		if total > total_max:
			temp = element
			total_max = total
		

	return temp







if __name__ == "__main__":

	X = np.array([[0], [1], [2], [7], [8], [9], [12], [14], [15]])
	X_2 = np.array([[1, 0], [7, 4], [9, 6], [2, 1], [4, 8], [0, 3], [13, 5], [6, 8], [7, 3], [3, 6], [2, 1], [8, 3], [10, 2], [3, 5], [5, 1], [1, 9], [10, 3], [4, 1], [6, 6], [2, 2]])
	K = 3
	# C = K_Means(X, K)
	# print(C)
	# C = K_Means(X_2, 2)
	C = K_Means_better(X, K)
	print("The result is:   \n\n", C)


	"1d 2 clusters"
	# y = [0,0,0,0,0]
	# x = [0, 1, 2, 7, 8]

	# y_2 = [0,0,0,0]
	# x_2 = [9, 12, 14, 15]

	# y_3 = [0]
	# x_3 = [4.5]
	# x_4= [13.2]
	# plt.scatter(x, y, color="red")
	# plt.scatter(x_2, y_2, color="blue")
	# plt.scatter(x_3, y_3, color="green")
	# plt.scatter(x_4, y_3, color="green")

	# plt.annotate("Center1", (4.5, 0))
	# plt.annotate("Center2", (13.2, 0))

	# plt.show()

	# y_3 = 1

	"1d 3 clusters"
	# y = [0,0,0]
	# x_1 = [0,1,2]
	# x_2 = [7,8,9]
	# x_3 = [12,14,15]

	# center_y = [0]
	# center1 = [1]
	# center2 = [9]
	# center3 = [14.5]

	# plt.scatter(x_1, y, color="red")
	# plt.scatter(x_2, y, color="blue")
	# plt.scatter(x_3, y, color="black")

	# plt.scatter(center1, center_y, color="green")
	# plt.scatter(center2, center_y, color="green")
	# plt.scatter(center3, center_y, color="green")

	# plt.annotate("Center1", (1, 0))
	# plt.annotate("Center2", (9, 0))
	# plt.annotate("Center3", (14.5, 0))

	# plt.show()



	"2d 2 clusters"

	# x = [1,7,9,2,4,0,13,6,7,3,2,8,10,3,5,1,10,4,6,2]
	# y = [0,4,6,1,8,3,5,8,3,6,1,3,2,5,1,9,3,1,6,2]\

	# x_2 = [4,13,6,7,8,10,10,6,7,9]
	# y_2 = [8, 5, 8, 3, 3, 2, 3, 6,4,6]

	# center_1_x = [3.00]
	# center_2_x = [8.00]

	# center_1_y = [3.00]
	# center_2_y = [5.00]

	# plt.scatter(x, y, color="red")
	# plt.scatter(x_2, y_2, color="blue")
	# plt.scatter(center_1_x, center_1_y, color="green")
	# plt.scatter(center_2_x, center_2_y, color="green")

	# plt.annotate("Center1", (3, 3))
	# plt.annotate("Center2", (8, 5))
	# plt.show()
	# print(len(x))


	"2d 3 clusters"

	# x_1 = [13,7,8,10,10,7,9]
	# y_1 = [5,3,3,2,3,4,6]

	# x_2 = [0,1,2,2,4,5]
	# y_2 = [3,0,1,2,1,1]

	# x_3 = [1,3,3,3,6,6]
	# y_3 = [9,5,6,8,6,8]

	# center_1_x = [3.83]
	# center_2_x = [2.29]
	# center_3_x = [9.14]

	# center_1_y = [7.00]
	# center_2_y = [1.29]
	# center_3_y = [3.71]

	# plt.scatter(x_1, y_1, color="red")
	# plt.scatter(x_2, y_2, color="blue")
	# plt.scatter(x_3, y_3, color="black")
	# plt.scatter(center_1_x, center_1_y, color="green")
	# plt.scatter(center_2_x, center_2_y, color="green")
	# plt.scatter(center_3_x, center_3_y, color="green")

	# plt.annotate("Center1", (3.83, 7.0))
	# plt.annotate("Center2", (2.29, 1.29))
	# plt.annotate("Center3", (9.14, 3.71))
	# plt.show()
