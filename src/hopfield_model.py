#!/usr/bin/env python3

import random
import numpy as np
import matplotlib.pyplot as plt


class HopfieldModel(object):

	def __init__(self):
		self.vt = np.empty((0, 0, 0))
		self.v0 = np.empty((0, 0, 0))
		self.v1 = np.empty((0, 0, 0))
		self.n = 0
		self.al = 0

	def import_images(self, image_paths):
		data = [np.loadtxt(path, delimiter=",").flatten() for path in image_paths]
		self.al = len(data)
		self.n = max([item.shape[0] for item in data])
		self.vt = np.zeros((self.al, self.n))
		for i, image in enumerate(data):
			self.vt[i] = image

	def normalise(self):
		self.vt = self.vt * 2 - 1

	def calc_weights(self):
		self.w = np.zeros((self.n, self.n))
		for i in range(self.n):
			for j in range(i, self.n):
				for k in range(self.al):
					self.w[i, j] = self.w[i, j] + self.vt[k, i] * self.vt[k, j]
				self.w[j, i] = self.w[i, j]
		self.w /= self.n

	def set_v0(self, icomp):
		self.v0 = self.vt[icomp]

	def add_noise(self, noise=0.1):
		self.v1 = self.v0
		for item in np.nditer(self.v1, op_flags=["readwrite"]):
			if random.random() > noise:
				item[...] = - item[...]
		plt.imshow(self.v1.reshape(20,20))
		plt.show()


	def run(self, itmax=100):
		for it in range(itmax):
			for i in range(self.n):
				sum = 0
				for j in range(self.n):
					sum += self.w[i, j] * self.v0[j]
					if sum > 0:
						self.v1[i] = 1
					else:
						if sum < 0:
							self.v1[i] = -1
						else:
							self.v1[i] = self.v0[i]
			if all(self.v0 == self.v1):
				break

		plt.imshow(self.v0.reshape(20,20))
		plt.show()
