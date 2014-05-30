#pass in a function and a subgradient
import random
import numpy

class Expression():
	def __init__(self, func, subgradient, hesVec, data):
		self.func = func;
		self.subgradient = subgradient
		self.hesVec = hesVec
		self.data = data
		self.numSamples = data.shape[0]
		self.numFeatures = data.shape[1]

	def get_value(self, w):
		obj = 0.0
		for i in xrange(self.numSamples):
			dataSample = numpy.transpose(self.data[i,:])
			obj = obj + self.func(w,dataSample)
		return -obj/self.numSamples

	def get_subgrad(self,w, batchSize):
		grad = numpy.zeros((self.numFeatures,1))
		sampleIndicies = random.sample(range(1,self.numSamples),batchSize)
		for i in xrange(batchSize):
			dataSample = numpy.transpose(self.data[i,:])
			grad = grad + self.subgradient(w,dataSample)
		return grad/batchSize

	def get_hesVec(self,w, batchSize):
		sampleIndicies = random.sample(range(1,self.numSamples),batchSize)
		output = numpy.zeros((self.numFeatures,1))
		for i in xrange(batchSize):
			dataSample = numpy.transpose(self.data[i,:])
			output = output + self.hesVec(w,dataSample)
		return output/batchSize

