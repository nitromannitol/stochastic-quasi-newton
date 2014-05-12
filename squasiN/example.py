#contains example of code 

#!/usr/bin/env python 

from Expression import Expression
from Problem import Problem
import numpy
import scipy as Sci
import scipy.linalg
import math
import random

#example function from paper
#create a matrix of examples 
#N is the number of features
N = 50
numSamples = 5000
Z = numpy.asarray([numpy.random.randint(0,1) for b in range(1,numSamples+1)])
X = numpy.matrix(numpy.random.random((numSamples,N)))

def binaryClassObjective(w):
	size = X.shape
	numExamples = size[0]
	obj = 0.0
	for i in xrange(numExamples):
		x = numpy.transpose(X[i,:])
		z = Z[i]
		cwx= 1.0/(1.0+math.exp(-numpy.transpose(x)*w))
		obj = obj + z*math.log(cwx) + (1.0-z)*math.log(1.0-cwx + 0.0001)
	return -obj/numExamples

def binaryClassObjectiveStochasticGradient(w, batchSize):
	size = X.shape
	numExamples = size[0]
	sampleIndicies = random.sample(range(1,numExamples),batchSize)
	grad = numpy.zeros((N,1))
	for i in sampleIndicies:
		x = numpy.transpose(X[i,:])
		z = Z[i]
		cwx= 1.0/(1.0+math.exp(-numpy.transpose(x)*w))
		grad = grad + (cwx - z)*(x)
	return grad/batchSize

def binaryClassObjectiveHessianVectorProduct(w,s, batchSize):
	size = X.shape
	numExamples = size[0]
	sampleIndicies = random.sample(range(1,numExamples),batchSize)
	#To do: not hardcode this
	output = numpy.zeros((N,1))
	for i in sampleIndicies:
		x = numpy.transpose(numpy.matrix((X[i,:])))
		cwx= 1.0/(1.0+math.exp(-numpy.transpose(x)*w))
		output = output + cwx*(1-cwx)*(numpy.transpose(x)*s).item()*x
	return output/batchSize


def test_expressions():
	w = numpy.transpose(numpy.matrix(numpy.random.random((1,N))))
	exp = Expression(binaryClassObjective, binaryClassObjectiveStochasticGradient, binaryClassObjectiveHessianVectorProduct, N)
	print(exp.get_value(w))
	print(exp.get_subgrad(w, 1))

def test_problems():
	exp = Expression(binaryClassObjective,binaryClassObjectiveStochasticGradient, binaryClassObjectiveHessianVectorProduct, N)
	prob = Problem(exp)
	print(prob.sgsolve())
	print(prob.sqnsolve())


if __name__ == "__main__":
	#test_expressions()
	test_problems()