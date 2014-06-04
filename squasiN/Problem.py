import numpy 
import scipy 
import matplotlib.pyplot as plt
import time
import math

#solves a given expression using stochastic quasi-newton 
class Problem():
	

	def __init__(self, exp, constraints = []):
		self.exp = exp
		self.constraints = constraints


	
	#Inputs: 
	#	x_0 : initial x
	#   M: size of the L-BFGS step computation
	#	L : compute correction pairs every L iterations
	#	K : number of iterations
	#	alpha : just use beta/niter for now can be a stepsize sequence  
	#	Z: Frequency at which we increase L
	#Outputs: final weights and optimum value 

	def sqnsolve(self, x_0 =None, M = 10, L = 30, K = 1000, alpha = None, gradientBatchSize = 5, hessianBatchSize = 300, verbose = False, Z =50):
		continuity_correction = 1e-10
	
		iterationsVal = numpy.zeros(K)
		averageTime = 0

		if x_0 is None:
			x_0 =  numpy.transpose(numpy.matrix(numpy.zeros((1,self.exp.paramSize))))
		x = x_0
		#average iterates
		x_av_j = x
		x_av_i = numpy.transpose(numpy.matrix(numpy.zeros((1,self.exp.paramSize))))


		#initialize curvature pairs 
		s_t = 0
		y_t = 0
		theta = 0

		#LFBGS parameters
		rho = numpy.ones((M,1))
		alp = numpy.ones((M,1))
		savedS = numpy.matrix(numpy.zeros((self.exp.paramSize, M)))
		savedY = numpy.matrix(numpy.zeros((self.exp.paramSize, M)))
		last = 0

		t = 0 #number of times the hessian has been updated
		prev = -999
		curr = 999
		numSame = 0

		for k in xrange(K):
			alpha = 0.2/math.sqrt((k+1))


			if(verbose is True):
				prev = curr
				curr = self.exp.get_value(x)
				iterationsVal[k] = curr
				#print k, curr
				if(math.fabs(prev - curr) < 1e-20):
					numSame+=1
				if(numSame > 50):
					break


			start = time.clock()

			#Calculate stochastic gradient
			stochastic_grad = self.exp.get_subgrad(x, gradientBatchSize)

			#update average iterate 
			x_av_i = x_av_i + x

			if k < 2*L:
				#stochastic gradient iteration 
				x = x - alpha*stochastic_grad
			else:
				#L-BFGS step computation with memory M 
				stepDirection = -stochastic_grad
				#compute uphill direction as follows
				index = last%M
				for i in xrange(min(M,t)):
					alp[index] = rho[index]*numpy.transpose(savedS[:,index])*stepDirection
					stepDirection = stepDirection - alp[index].item()*savedY[:,index]
					index = M - ((-index+1) % M) -1

				stepDirection = theta*stepDirection

				for i in xrange(min(M,t)):
					index = (index+1)%M
					b = rho[index]*numpy.transpose(savedY[:,index])*stepDirection
					b = b.item()
					stepDirection = stepDirection + (alp[index].item() - b)*savedS[:,index]

				#update x with the step direction 
				x = x + alpha*stepDirection


			if k%L == 0:
				t = t +1
				x_av_i = x_av_i/L
				#compute new curvature pairs 
				s_t = x_av_i - x_av_j
				y_t = self.exp.get_hesVec(x_av_i,hessianBatchSize, s_t)
				x_av_j = x_av_i
				x_av_i = numpy.transpose(numpy.matrix(numpy.zeros((1,self.exp.paramSize))))
				#save theta so we don't compute it multiple times
				ys = numpy.transpose(s_t)*y_t + continuity_correction
				theta = (ys/(( numpy.transpose(y_t)*y_t + continuity_correction))).item()
				last +=1
				if M > 0:
					last = last%M
					savedS[:,last] = s_t
					savedY[:,last] = y_t
					rho[last] = float(1)/ys
			end = time.clock() 
			if(k%Z == 0):
				L+=1
			averageTime+= end - start

		averageTime/=K

		if(verbose is True):
			plt.plot(iterationsVal[1:k], label = "Stochastic Quasi-Newton")
			plt.xlabel('Iterations')
			plt.ylabel('Objective Value')
			plt.legend(loc='lower right')
			#plt.show()
			fig = plt.figure(1)
			# We define a fake subplot that is in fact only the plot.  
			plot = fig.add_subplot(111)
			# We change the fontsize of minor ticks label 
			plot.tick_params(axis='both', which='major', labelsize=14)
			plot.tick_params(axis='both', which='minor', labelsize=12)
			print ('Total number of times Hessian was updated: ', t)
			print 'We converged in ', k+1, ' iterations'
			print 'The average iteration took ', averageTime, ' seconds'
			#print iterationsVal

		return (self.exp.get_value(x), x, averageTime)


	#basic non-robust implementation of stochastic gradient descent
	def sgsolve(self, K = 1000, gradientBatchSize = 50, verbose = False):
		x = numpy.transpose(numpy.matrix(numpy.zeros((1,self.exp.paramSize))))

		iterationsVal = numpy.zeros(K)
		averageTime = 0
		prev = -999
		curr = 999
		numSame = 0

		for k in xrange(K):	

			if(verbose is True):
				prev = curr
				curr = self.exp.get_value(x)
				iterationsVal[k] = curr
				#print k, curr
				if(math.fabs(prev - curr) < 1e-20):
					numSame+=1
				if(numSame > 50):
					break


			start = time.clock()
			alpha = 0.1/math.sqrt((k+1))

			#Calculate stochastic gradient
			stochastic_grad = self.exp.get_subgrad(x, gradientBatchSize)
			#print stochastic_grad

			#Perform step update
			x = x-alpha*stochastic_grad
			end = time.clock()

			averageTime+= end - start

		averageTime/=K

			
		
		if(verbose is True):

			plt.plot(iterationsVal, label = "Stochastic Gradient Descent")
			plt.xlabel('Iterations')
			plt.ylabel('Objective Value')
			print 'The average iteration took ', averageTime, ' seconds'
			print 'We converged in ', k+1, ' iterations'
			#plt.show()


		return (self.exp.get_value(x), x, averageTime)





