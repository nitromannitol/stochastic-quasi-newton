#pass in a function and a subgradient
class Expression():
	def __init__(self, func, subgradient):
		self.func = func;
		self.subgradient = subgradient
	def get_value(self, x):
		return self.func(x)
	def get_subgrad(self,x):
		return self.subgradient(x)
