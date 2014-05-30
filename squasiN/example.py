#contains example of code 

#!/usr/bin/env python 

from Expression import Expression
from Problem import Problem
import numpy
import scipy as Sci
import scipy.linalg
import math
import random


#evalutes the objective on a data point, the data is of the form [label, x_1, .., x_p]
#where p is the number of features 
def logisticObjective(w, dataSample):
	continuity_correction = 1e-10
	label = dataSample[0].item()
	x = dataSample[1:]
	cwx= 1.0/(1.0+math.exp(-numpy.transpose(x)*w))
	return label*math.log(cwx) + (1.0-label)*math.log(1.0-cwx + continuity_correction)

#the logistic gradient of one data vector
def logisticGradient(w, dataSample):
	label = dataSample[0].item()
	x = dataSample[1:]
	cwx= 1.0/(1.0+math.exp(-numpy.transpose(x)*w))
	return (cwx - label)*(x)

#the logistic hessian vector of one data sample and vector s
def logisticHessianVec(w,dataSample,s):
	x = dataSample[1:]
	cwx= 1.0/(1.0+math.exp(-numpy.transpose(x)*w))
	return cwx*(1-cwx)*(numpy.transpose(x)*s).item()*x

def test_expressions():
	w = numpy.transpose(numpy.matrix(numpy.random.random((1,numFeatures))))
	exp = Expression(binaryClassObjective, binaryClassObjectiveStochasticGradient, binaryClassObjectiveHessianVectorProduct, numFeatures)
	print(exp.get_value(w))
	print(exp.get_subgrad(w, 1))

def test_problems():
	#generate synthetic data
	numFeatures = 50
	numSamples = 7000

	mean0 = 0
	mean1 = 10

	Z0 = numpy.matrix([0 for b in range(1,numSamples/2 + 1)])
	X0 = numpy.random.normal(loc = mean0, size = (numSamples/2, numFeatures))
	data0 = numpy.hstack([numpy.transpose(Z0), X0])
	Z1 = numpy.matrix([1 for b in range(1,numSamples/2 + 1)])
	X1 = numpy.random.normal(loc = mean1, size = (numSamples/2, numFeatures))
	data1 = numpy.hstack([numpy.transpose(Z1), X1])

	data = numpy.vstack([data0,data1])



	exp = Expression(logisticObjective,logisticGradient, logisticHessianVec, data, numFeatures)
	prob = Problem(exp, constraints = [])
	print(prob.sgsolve(verbose = True, K = 100, gradientBatchSize = 50))
	print(prob.sqnsolve(verbose = True, K = 100, gradientBatchSize = 50, hessianBatchSize = 300))


if __name__ == "__main__":
	#test_expressions()
	test_problems()