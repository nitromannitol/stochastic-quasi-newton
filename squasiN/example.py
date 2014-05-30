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
	continuity_correction = 1e-50
	label = dataSample[0].item()
	x = dataSample[1:]
	z = -numpy.transpose(x)*w
	if(math.fabs(z) > 600):
		z = 600
	cwx= 1.0/(1.0+math.exp(z))
	return label*math.log(cwx) + (1.0-label)*math.log(1.0-cwx + continuity_correction)

#the logistic gradient of one data vector
def logisticGradient(w, dataSample):
	label = dataSample[0].item()
	x = dataSample[1:]
	z = -numpy.transpose(x)*w
	if(math.fabs(z) > 600):
		z = 600
	cwx= 1.0/(1.0+math.exp(z))
	return (cwx - label)*(x)

#the logistic hessian vector of one data sample and vector s
def logisticHessianVec(w,dataSample,s):
	x = dataSample[1:]
	z = -numpy.transpose(x)*w
	if(math.fabs(z) > 600):
		z = 600
	cwx= 1.0/(1.0+math.exp(z))
	return cwx*(1-cwx)*(numpy.transpose(x)*s).item()*x

#Calculates the logistic error of parameter x_opt on the 
#data set data 
def logisticError(x_opt,data):
	numWrong = 0
	numSamples = data.shape[0]
	for i in xrange(1,numSamples):
		dataSample = numpy.transpose(data[i,:])
		label = dataSample[0].item()
		x = dataSample[1:]
		z = numpy.transpose(x)*x_opt
		#prevent overflow
		if(math.fabs(z) > 600):
			z = 600
		prob = math.exp(z)/(1+ math.exp(z))
		if(prob > 0.5):
			pred = 1
		else:
			pred = 0
		if(pred > label):
			numWrong+=1 
	return (float(numWrong)/float(numSamples))*100

def test_expressions():
	w = numpy.transpose(numpy.matrix(numpy.random.random((1,numFeatures))))
	exp = Expression(binaryClassObjective, binaryClassObjectiveStochasticGradient, binaryClassObjectiveHessianVectorProduct, numFeatures)
	print(exp.get_value(w))
	print(exp.get_subgrad(w, 1))

def test_problems():
	#generate synthetic data
	numFeatures = 100
	numSamples = 7000

	mean0 = 0
	mean1 = 10

	std0 = 1
	std1 = 1

	Z0 = numpy.matrix([0 for b in range(1,numSamples/2 + 1)])
	X0 = numpy.random.normal(loc = mean0, scale = std0, size = (numSamples/2, numFeatures))
	data0 = numpy.hstack([numpy.transpose(Z0), X0])
	Z1 = numpy.matrix([1 for b in range(1,numSamples/2 + 1)])
	X1 = numpy.random.normal(loc = mean1, scale = std1, size = (numSamples/2, numFeatures))
	data1 = numpy.hstack([numpy.transpose(Z1), X1])

	data = numpy.vstack([data0,data1])

	exp = Expression(logisticObjective,logisticGradient, logisticHessianVec, data, numFeatures)
	prob = Problem(exp, constraints = [])
	[optval, x_opt] =  prob.sgsolve(verbose = False, K = 500, gradientBatchSize = 50)
	print 'Error of SGD is ' +  str(logisticError(x_opt, data) )+ '%'
	print optval, x_opt


	[optval, x_opt] =  prob.sqnsolve(verbose = False, K = 500, gradientBatchSize = 5, hessianBatchSize = 300)
	print 'Error of SQN is ' +  str(logisticError(x_opt, data) )+ '%'
	print optval, x_opt



if __name__ == "__main__":
	#test_expressions()
	test_problems()