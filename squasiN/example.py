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
	#generate data
	numFeatures = 50
	numSamples = 5000
	Z = numpy.matrix([numpy.random.randint(0,1) for b in range(1,numSamples+1)])
	X = numpy.matrix(numpy.random.random((numSamples,numFeatures)))
	data = numpy.hstack([numpy.transpose(Z),X])
	exp = Expression(logisticObjective,logisticGradient, logisticHessianVec, data, numFeatures)
	prob = Problem(exp)
	print(prob.sgsolve())
	print(prob.sqnsolve())


if __name__ == "__main__":
	#test_expressions()
	test_problems()