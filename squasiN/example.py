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
N = 500
M = 5000
Z = numpy.asarray([numpy.random.randint(0,1) for b in range(1,M+1)])
X = numpy.random.random((N,M))

def binaryClassObjective(w):
	size = X.shape
	numExamples = size[0]
	obj = 0.0
	for i in xrange(numExamples):
		x = numpy.transpose(numpy.matrix((X[i,:])))
		z = Z[i]
		cwx= 1.0/(1.0+math.exp(-numpy.transpose(x)*w))
		obj = obj + z*math.log(cwx) + (1.0-z)*math.log(1.0-cwx + 0.0001)
	return -obj/numExamples

def binaryClassObjectiveStochasticGradient(w, batchSize = 10):
	size = X.shape
	numExamples = size[0]
	sampleIndicies = random.sample(range(1,numExamples+1),batchSize)
	grad = 0
	for i in sampleIndicies:
		x = numpy.transpose(numpy.matrix((X[i,:])))
		z = Z[i]
		cwx= 1.0/(1.0+math.exp(-numpy.transpose(x)*w))
		grad = grad + (cwx - z)*(x)
	return grad



# example functions and their subgradients 
#f(x) = max_{i=1,..m} (a_i^tx + b_i)
numpy.random.seed(1)
A =  numpy.random.random((3, 3))
bVec = numpy.random.random((3,1))
def function1(x):
	return max(A*x+bVec).item()
def subgrad1(x, batchsize = 1):
	vals = A*x+bVec
	return A[numpy.argmax(vals) ,:]

#L_1 norm
def function2(x):
	return scipy.linalg.norm(x,ord = 1)
def subgrad2(x, batchsize = 1):
	zeroVals = (x == 0)
	x[zeroVals] = 1
	return numpy.sign(x)

#maximum eigenvalue of a symmetric matrix
N = 100
M = 5
bArray = []
numpy.random.seed(1)
for i in xrange(M):
	b = numpy.random.random_integers(-2000,2000,size=(N,N))
	b_symm = (b + b.T)/2
	bArray.append(b_symm)

#x is 4 dimensional 
def function3(x):
	B = bArray[0]
	for i in xrange(1,M):
		B += (x[i-1].item())*bArray[i]		
	eigenValues,eigenVectors = scipy.linalg.eig(B)
	return(max(eigenValues))


def subgrad3(x, batchsize = 1):
	B = bArray[0]
	for i in xrange(1,M):
		B += (x[i-1].item())*bArray[i]		
	eigenValues,eigenVectors = scipy.linalg.eig(B)
	idx = eigenValues.argsort()[::-1]
	eigenVectors = eigenVectors[:,idx]
	maxEigenVector = eigenVectors[:,0]
	#make the array 
	gradient = numpy.array([])
	for i in xrange(M):
		B = bArray[i];
		gradient = numpy.append(gradient, numpy.transpose(maxEigenVector)*B*maxEigenVector)
	return gradient


def test_expressions():
	exp1 = Expression(function1, subgrad1)
	x = numpy.transpose(numpy.matrix([1, 2, 3]))
	print exp1.get_value(x)
	print exp1.get_subgrad(x)
	exp2 = Expression(function2, subgrad2)
	print(exp2.get_value(x))
	print(exp2.get_subgrad(x))
	x2 = numpy.transpose(numpy.matrix([1, 2, 3, 4]))
	exp3 = Expression(function3, subgrad3)
	print(exp3.get_value(x2))
	print(exp3.get_subgrad(x2))
	w = numpy.transpose(numpy.matrix(numpy.random.random((1,5000))))
	exp = Expression(binaryClassObjective, binaryClassObjectiveStochasticGradient)
	print(exp.get_value(w))
	print(exp.get_subgrad(w))

def test_problems():
	exp = Expression(binaryClassObjective,binaryClassObjectiveStochasticGradient)
	prob = Problem(exp)
	print(prob.solve())


if __name__ == "__main__":
	test_expressions()
	test_problems()