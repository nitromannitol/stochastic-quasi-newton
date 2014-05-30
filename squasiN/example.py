#contains example of code 

#!/usr/bin/env python 

from Expression import Expression
from Problem import Problem
import numpy
import scipy as Sci
import scipy.linalg
import math
import random
from sklearn.linear_model import SGDClassifier


#evalutes the objective on a data point, the data is of the form [label, x_1, .., x_p]
#where p is the number of features 
def logisticObjective(w, dataSample):
	continuity_correction = 1e-20
	label = dataSample[0].item()
	x = dataSample[1:]
	z = (numpy.transpose(w)*x).item()
	if(z > 600):
		z = 600
	if(z < -600):
		z = -600
	sigmoid = 1/(1 + math.exp(-z))
	return label*math.log(sigmoid + continuity_correction) + (1.0-label)*math.log(1.0 - sigmoid + continuity_correction)

#the logistic gradient of one data vector
#since we are doing gradient ascent, this is negative
def logisticGradient(w, dataSample):
	label = dataSample[0].item()
	x = dataSample[1:]
	z = (numpy.transpose(w)*x).item()
	if(z > 600):
		z = 600
	if(z < -600):
		z = -600
	sigmoid = 1/(1 + math.exp(-z))
	return-((label-sigmoid))*x

#the logistic hessian vector of one data sample and vector s
def logisticHessianVec(w,dataSample,s):
	x = dataSample[1:]
	z = (-numpy.transpose(x)*w).item()
	if(z > 600):
		z = 600
	if(z < -600):
		z = -600
	sigmoid = 1/(1 + math.exp(-z))
	return sigmoid*(1-sigmoid)*(numpy.transpose(x)*s).item()*x

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
		if(z > 600):
			z = 600
		if(z < -600):
			z = -600
		prob = math.exp(z)/(1+ math.exp(z))
		if(prob > 0.5):
			pred = 1
		else:
			pred = 0
		if(pred != label):
			numWrong+=1 
		#if(pred == 1):
		#	print('-----')
		#if(pred == 0):
		#	print('o')
	return (float(numWrong)/float(numSamples))*100

def test_expressions():
	w = numpy.transpose(numpy.matrix(numpy.random.random((1,numFeatures))))
	exp = Expression(binaryClassObjective, binaryClassObjectiveStochasticGradient, binaryClassObjectiveHessianVectorProduct, numFeatures)
	print(exp.get_value(w))
	print(exp.get_subgrad(w, 1))


def test_problems():

	data = numpy.fliplr(numpy.matrix(numpy.loadtxt('spam.data')))
	numFeatures = data.shape[1] -1
	numSamples = data.shape[0]


	#test on skilearn stochastic gradient descent
	y = numpy.ravel(numpy.array(data[:,0]))
	X = data
	X = numpy.array(numpy.delete(X,0,axis = 1))
	clf = SGDClassifier(loss="log", penalty="l2")
	clf.fit(X,y)
	error = sum(y - clf.predict(X))/len(y)

	p = numpy.transpose(numpy.matrix(clf.coef_))
	print 'Error of SKILearn SGD is ' +  str(logisticError(p, data) )+ '%'



	numDesiredSamples = 10000
	choices = numpy.random.choice(xrange(numSamples), size = numDesiredSamples)

	dataLarge = numpy.zeros((numSamples, numFeatures+1))

	numpy.random.seed(42)
	for i in xrange(numSamples):
		choice = choices[i]
		dataLarge[i,:] = data[choice,:]

	data = numpy.matrix(dataLarge)


	


	exp = Expression(logisticObjective,logisticGradient, logisticHessianVec, data, numFeatures)
	prob = Problem(exp, constraints = [])
	[optval, x_opt] =  prob.sgsolve(verbose = False, K = 25, gradientBatchSize = 50)
	print 'Error of SGD is ' +  str(logisticError(x_opt, data) )+ '%'
	#print optval, x_opt


	[optval, x_opt] =  prob.sqnsolve(verbose = False, K = 25, gradientBatchSize = 5, hessianBatchSize = 300)
	print 'Error of SQN is ' +  str(logisticError(x_opt, data) )+ '%'
	#print optval, x_opt



if __name__ == "__main__":
	#test_expressions()
	test_problems()