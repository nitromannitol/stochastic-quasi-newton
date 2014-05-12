#pass in a function and a subgradient
class Expression():
	def __init__(self, func, subgradient, hesVec, inputSize):
		self.func = func;
		self.subgradient = subgradient
		self.hesVec = hesVec
		self.inputSize = inputSize
	def get_value(self, x):
		return self.func(x)
	def get_subgrad(self,x, batchSize):
		return self.subgradient(x, batchSize)
	def get_hesVec(self,x,s, batchSize):
		return self.hesVec(x,s, batchSize)
