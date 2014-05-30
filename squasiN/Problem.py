import numpy 
import scipy 
import matplotlib.pyplot as plt
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
	#Outputs: final weights and optimum value 

	def sqnsolve(self, x_0 =None, M = 10, L = 20, K = 1000, alpha = None, gradientBatchSize = 20, hessianBatchSize = 20, verbose = False):
		continuity_correction = 1e-10
	
		iterationsVal = numpy.zeros(K)


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

		for k in xrange(K):
			alpha = 5/(k+1)

			if(verbose is True):
				iterationsVal[k] = self.exp.get_value(x)
				print k, iterationsVal[k]


			#Calculate stochastic gradient
			stochastic_grad = self.exp.get_subgrad(x, gradientBatchSize)

			#update average iterate 
			x_av_i = x_av_i + x

			if k < 2*L:
				#stochastic gradient iteration 
				x = x - alpha*stochastic_grad
			else:
				#L-BFGS step computation with memory M 
				stepDirection = stochastic_grad
				#compute uphill direction as follows
				index = last
				for i in xrange(min(M,t)):
					alp[index] = rho[index]
					stepDirection = stepDirection - alp[index].item()*savedY[:,index]
					index = M - ((-index + 1) % M)
				stepDirection = theta*stepDirection
				for i in xrange(min(M,t)):
					index = index%M
					b = rho[index]*numpy.transpose(savedY[:,index])*stepDirection
					b = b.item()
					stepDirection = stepDirection + (alp[index].item() - b)*savedS[:,index]

				#update x with the step direction 
				x = x - alpha*stepDirection

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
				if M > 0:
					last = last%M
					savedS[:,last] = s_t
					savedY[:,last] = y_t
					rho[last] = 1/ys

		if(verbose is True):
			plt.plot(iterationsVal)
			plt.xlabel('Iterations')
			plt.ylabel('Objective Value')
			plt.show()

		return (self.exp.get_value(x), x)


	#basic non-robust implementation of stochastic gradient descent
	def sgsolve(self, K = 1000, gradientBatchSize = 10, verbose = False):
		x = numpy.transpose(numpy.matrix(numpy.zeros((1,self.exp.paramSize))))
		

		iterationsVal = numpy.zeros(K)


		for k in xrange(K):
			alpha = 4/(k+1)

			#Calculate stochastic gradient
			stochastic_grad = self.exp.get_subgrad(x, gradientBatchSize)

			#Perform step update
			x = x-alpha*stochastic_grad

			if(verbose is True):
				iterationsVal[k] = self.exp.get_value(x)
				print k, iterationsVal[k]
		
		if(verbose is True):
			plt.plot(iterationsVal)
			plt.xlabel('Iterations')
			plt.ylabel('Objective Value')
			plt.show()


		return (self.exp.get_value(x), x)





