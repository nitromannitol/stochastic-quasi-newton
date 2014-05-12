#pass in a function and a subgradient
import random
import numpy

class Expression():
	def __init__(self, func, subgradient, hesVec, data, labels):
		self.func = func;
		self.subgradient = subgradient
		self.hesVec = hesVec
		self.data = data
		self.labels = labels
		self.numSamples = data.shape[0]
		self.numFeatures = data.shape[1]

	def get_value(self, w):
		obj = 0.0
		for i in xrange(self.numSamples):
			x = numpy.transpose(self.data[i,:])
			z = self.labels[i]
			obj = obj + self.func(w,x,z)
		return -obj/self.numSamples

	def get_subgrad(self,w, batchSize):
		grad = numpy.zeros((self.numFeatures,1))
		sampleIndicies = random.sample(range(1,self.numSamples),batchSize)
		for i in xrange(batchSize):
			x = numpy.transpose(self.data[i,:])
			z = self.labels[i]
			grad = grad + self.subgradient(w,x,z)
		return grad/batchSize

	def get_hesVec(self,w,s, batchSize):
		sampleIndicies = random.sample(range(1,self.numSamples),batchSize)
		output = numpy.zeros((self.numFeatures,1))
		for i in xrange(batchSize):
			x = numpy.transpose(self.data[i,:])
			output = output + self.hesVec(w,s,x)
		return output/batchSize

