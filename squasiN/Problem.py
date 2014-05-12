import numpy 
import scipy 
#solves a given expression using stochastic quasi-newton 
class Problem():
	def __init__(self, exp, constraints = []):
		self.exp = exp
		self.constraints = constraints

	#does quasi newton on the expression
	#returns the optimum value and the current point
	#we can call self.exp.get_subgrad(currPoint)
	#to get the subgradient at that point
	
	#Inputs: 
	#	x_0 : initial x
	#   M: size of the L-BFGS step computation
	#	L : compute correction pairs every L iterations
	#	K : number of iterations
	#	alpha : just use 1/niter for now can be a stepsize sequence  
	#Outputs: final weights

	def sqnsolve(self, x_0 =None, M = 20, L = 20, K = 1000, alpha = None, gradientBatchSize = 20, hessianBatchSize = 20):
		if x_0 is None:
			x_0 =  numpy.transpose(numpy.matrix(numpy.zeros((1,self.exp.inputSize))))
		x = x_0
		#average iterates
		x_av_j = x
		x_av_i = numpy.transpose(numpy.matrix(numpy.zeros((1,self.exp.inputSize))))


		#initialize curvature pairs 
		s_t = 0
		y_t = 0
		theta = 0

		#LFBGS parameters
		rho = numpy.ones((M,1))
		alp = numpy.ones((M,1))
		delW = numpy.matrix(numpy.zeros((self.exp.inputSize, M)))
		delG = numpy.matrix(numpy.zeros((self.exp.inputSize, M)))
		last = 0

		t = 0 #number of times the hessian has been updated

		for k in xrange(K):
			alpha = 1/(k+1)

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
					stepDirection = stepDirection - alp[index].item()*delG[:,index]
					index = M - ((-index + 1) % M)
				stepDirection = theta*stepDirection
				for i in xrange(min(M,t)):
					index = index%M
					b = rho[index]*numpy.transpose(delG[:,index])*stepDirection
					b = b.item()
					stepDirection = stepDirection + (alp[index].item() - b)*delW[:,index]

				#update x with the step direction 
				x = x - alpha*stepDirection

			if k%L == 0:				
				t = t +1
				x_av_i = x_av_i/L
				#compute new curvature pairs 
				s_t = x_av_i - x_av_j
				y_t = self.exp.get_hesVec(x_av_i, s_t, hessianBatchSize)
				x_av_j = x_av_i
				x_av_i = numpy.transpose(numpy.matrix(numpy.zeros((1,self.exp.inputSize))))
				#save theta so we don't compute it multiple times
				ys = numpy.transpose(s_t)*y_t + 0.0000001
				theta = (ys/(( numpy.transpose(y_t)*y_t + 0.0000001))).item()
				if M > 0:
					last = last%M
					delW[:,last] = s_t
					delG[:,last] = y_t
					rho[last] = 1/ys


		return (self.exp.get_value(x), x)


	#basic non-robust implementation of stochastic gradient descent
	def sgsolve(self, K = 1000, gradientBatchSize = 10):
		x = numpy.transpose(numpy.matrix(numpy.zeros((1,self.exp.inputSize))))

		for k in xrange(K):
			alpha = 1/(k+1)

			#Calculate stochastic gradient
			stochastic_grad = self.exp.get_subgrad(x, gradientBatchSize)

			#Perform step update
			x = x-alpha*stochastic_grad


		return (self.exp.get_value(x), x)





