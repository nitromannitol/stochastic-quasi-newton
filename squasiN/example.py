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


#evalutes the likelihood on a data point, the data is of the form [label, x_1, .., x_p]
#where p is the number of features 
def logLikelihood(w, dataSample):
	continuity_correction = 1e-10
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
	return((label-sigmoid))*x

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

def totalObjective(x_opt, data):
	obj = 0
	numSamples = data.shape[0]
	for i in xrange(1,numSamples):
		dataSample = numpy.transpose(data[i,:])
		obj += logLoss(x_opt,dataSample)
	return obj




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
	return (float(numWrong)/float(numSamples))*100






def test_expressions():
	w = numpy.transpose(numpy.matrix(numpy.random.random((1,numFeatures))))
	exp = Expression(binaryClassObjective, binaryClassObjectiveStochasticGradient, binaryClassObjectiveHessianVectorProduct, numFeatures)
	print(exp.get_value(w))
	print(exp.get_subgrad(w, 1))


def test_various_parameters(prob, exp, data, verbosity):
	#Run our implementation of stochastic gradient descent
	print('Testing SGD')
	[optval, x_opt, sgTime] =  prob.sgsolve(verbose = False, K = 300, gradientBatchSize = 50)
	print 'We ran SGD with gradient batch size', 50
	print 'Objective is', optval
	print 'Error of SGD is ' +  str(logisticError(x_opt, data))+ '%'
	print optval, x_opt
	sgdOpt = optval


	Gbs = (1,30,500)
	Hbs = (1,300, 1000)
	Ms = (10,20,100)
	Ls = (1,30,85)


	for Gb in Gbs: 
		[optval, x_opt, itertime]  =  prob.sqnsolve(verbose = verbosity, K = 300, hessianBatchSize = 300, gradientBatchSize = Gb, L = 20)
		print 'We ran SQN with gradient batch size, ', Gb
		print 'Relative Objective is ', optval/sgdOpt
		print 'Relative Time is, ', itertime/sgTime
		print 'Error of SQN is ' +  str(logisticError(x_opt, data))+ '%'
		print '----'

	for Hb in Hbs:
		#Run our implementation of stochastic quasi newton gradient descent
		[optval, x_opt, itertime] =  prob.sqnsolve(verbose = verbosity, K = 300, hessianBatchSize = Hb, gradientBatchSize = 30, L = 20)
		print 'We ran SQN with hessian batch size, ', Hb
		print 'Relative Objective is ', optval/sgdOpt
		print 'Relative Time is, ', itertime/sgTime
		print 'Error of SQN is ' +  str(logisticError(x_opt, data))+ '%'
		print '----'
		#print optval, x_opt

	for Mval in Ms:
		[optval, x_opt, itertime]  =  prob.sqnsolve(verbose = verbosity, K = 300, hessianBatchSize = 300, gradientBatchSize = 30, L = 20, M = Mval)
		print 'We ran SQN with M: ', Mval
		print 'Relative Time is, ', itertime/sgTime
		print 'Relative Objective is ', optval/sgdOpt
		print 'Error of SQN is ' +  str(logisticError(x_opt, data))+ '%'
		print '----'


	for Lval in Ls:
		[optval, x_opt, itertime] =  prob.sqnsolve(verbose = verbosity, K = 300, hessianBatchSize = 300, gradientBatchSize = 30, L = Lval)
		print 'We ran SQN with L: ', Lval
		print 'Relative Objective is ', optval/sgdOpt
		print 'Relative Time is, ', itertime/sgTime
		print 'Error of SQN is ' +  str(logisticError(x_opt, data))+ '%'
		print '----'



def test_problems():

	#data = numpy.fliplr(numpy.matrix(numpy.loadtxt('spam.data')))
	

	#read in the data, ignoring the header row 
	data = numpy.matrix(numpy.genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:])
	numFeatures = data.shape[1] -1
	numSamples = data.shape[0]


	#note that this artificially boosts performance of SGD 
	#over standard gradient descent 
	#numDesiredSamples = 10000
	#choices = numpy.random.choice(numSamples, size = numDesiredSamples)

	#dataLarge = numpy.zeros((numDesiredSamples, numFeatures+1))

	#numpy.random.seed(55)
	#for i in xrange(numDesiredSamples):
	#	choice = choices[i]
	#	dataLarge[i,:] = data[choice,:]

	#data = numpy.matrix(dataLarge)


	exp = Expression(logLikelihood,logisticGradient, logisticHessianVec, data, numFeatures)
	prob = Problem(exp, constraints = [])

	#test on skilearn stochastic gradient descent
	#y = numpy.ravel(numpy.array(data[:,0]))
	#X = data
	#X = numpy.array(numpy.delete(X,0,axis = 1))
	#clf = SGDClassifier(loss="log", penalty=None, shuffle = True, warm_start = False, n_iter = 500)
	#clf.fit(X,y)
	#error = float(numpy.count_nonzero( y!= clf.predict(X)))/len(y)


	#p = numpy.transpose(numpy.matrix(clf.coef_))
	#print exp.get_value(p)

	#print 'Error of SKILearn SGD is ' +  str(logisticError(p, data) )+ '%'
	#print error 
	#print 'Objective of SKILearn SGD is ', + str(totalObjective(p, data)) 


	#optimal is 12.4858197

	#for i in xrange(10):
	#[val, x, time] = prob.sgsolve(verbose = True, K = 500, gradientBatchSize  = 50)
	#[val, x, time] = prob.sqnsolve(verbose = True, K= 500, gradientBatchSize = 30, L = 30, Z= 10000)
	test_various_parameters(prob, exp, data, True)


	



if __name__ == "__main__":
	#test_expressions()
	test_problems()