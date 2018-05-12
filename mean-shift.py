import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

from sklearn.datasets.samples_generator import make_blobs

#centers = [[2, 2], [8, 6], [8, 0], [2, 9]]
X, y = make_blobs(n_samples=30, cluster_std=1, shuffle=True, centers=4, n_features=8, random_state=0)
'''
X = np.array([
		[1, 2],
		[1.5, 1.8],
		[5, 8],
		[8, 8],
		[1, 0.6],
		[5, 9] ,
		[8, 6] ,
		[7, 7] ,
		[3, 3] ,
		[2, 4] ,
		[2, 0] ,
		[8, 1] ,
		[9.2, 0.5] ,
		[10, 2] ])
'''

#plt.scatter(X[:, 0], X[:, 1], s=100)
#plt.show()

colors = 10*["g", "r", "c", "b", "k"]


class MeanShift:
	def __init__(self, radius=None, radius_norm_step=100):
		self.radius = radius
		self.radius_norm_step = radius_norm_step

	def fit(self, data):

		if self.radius == None:
			all_data_centroid = np.average(data, axis=0)
			all_data_norm = np.linalg.norm(all_data_centroid)
			self.radius = all_data_norm / self.radius_norm_step

		centroids = {}
		for i in range(len(data)):
			centroids[i] = data[i]

		weights = [i**2 for i in range(self.radius_norm_step)][::-1]

		while True:
			new_centroids = []
			for i in centroids:
				in_bandwidth = []
				centroid = centroids[i]

				for featureset in data:
					distance = np.linalg.norm(featureset - centroid)
					if distance == 0:
						distance = 1e-10
					weight_index = int(distance/self.radius)
					if weight_index > (self.radius_norm_step - 1):
						weight_index = self.radius_norm_step - 1
					to_add = (weights[weight_index])*[featureset] # to improve
					in_bandwidth += to_add

				new_centroid = np.average(in_bandwidth, axis=0)
				new_centroids.append(tuple(new_centroid))

			uniques = sorted(list(set(new_centroids)))


			to_pop = []
			for i in uniques:
				if i in to_pop:
					continue
				for ii in uniques:
					if i == ii or ii in to_pop:
						pass
					elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius:
						to_pop.append(ii)
						break

			[uniques.remove(i) for i in to_pop]

			prev_centroids = dict(centroids)

			centroids = {}
			for i in range(len(uniques)):
				centroids[i] = np.array(uniques[i])

			optimized = True
			for i in centroids:
				if not np.array_equal(centroids[i], prev_centroids[i]):
					optimized = False
					break

			if optimized:
				break

		self.centroids = centroids

		self.classifications = {}
		for i in range(len(self.centroids)):
			self.classifications[i] = []

		for featureset in data:
			distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
			classification = distances.index(min(distances))
			self.classifications[classification].append(featureset)

	def predict(self, data):
		distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification

clf = MeanShift()
clf.fit(X)

centroids = clf.centroids
print("{} centroids found.".format(len(centroids)))
for classification in  clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0], featureset[1], s=50, marker="x", color=color, linewidth=2)

for c in centroids:
	plt.scatter(centroids[c][0], centroids[c][1], s=70, marker='*', color='k')
plt.show()













