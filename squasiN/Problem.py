import numpy 
import scipy 
#solves a given expression using stochastic quasi-newton 
class Problem():
	def __init__(self, exp, constraints):
		self.exp = exp
		self.constraints = constraints

	#does quasi newton on the expression
	#returns the optimum value and the current point
	#we can call self.exp.get_subgrad(currPoint)
	#to get the subgradient at that point
	def solve(self, stepsize = 1/100, batchSize = 20):
		return (0,0)


