#pass in a function and a subgradient
import random
import numpy
from multiprocessing import Pool


class Expression():
	def __init__(self, func, subgradient, hesVec, data, paramSize):
		self.func = func;
		self.subgradient = subgradient
		self.hesVec = hesVec
		self.data = data
		self.numSamples = data.shape[0]
		self.paramSize = paramSize

	def get_value(self, w):
		obj = 0.0
		datatranspose = self.data.T
		for i in xrange(self.numSamples):
			obj = obj + self.func(w, datatranspose[:,i])
		return -obj/self.numSamples

	def get_subgrad(self,w, batchSize):
		grad = numpy.zeros((self.paramSize,1))
		sampleIndicies = random.sample(range(1,self.numSamples),batchSize)
		datatranspose = self.data.T
		for i in sampleIndicies:
			grad = grad + self.subgradient(w,datatranspose[:,i])
		return grad/batchSize

	def get_hesVec(self,w, batchSize, s):
		sampleIndicies = random.sample(range(1,self.numSamples),batchSize)
		output = numpy.zeros((self.paramSize,1))
		datatranspose = self.data.T
		for i in sampleIndicies:
			output = output + self.hesVec(w,datatranspose[:,i], s)
		return output/batchSize

