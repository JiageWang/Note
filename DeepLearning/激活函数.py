import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

class Sigmoid(object):
	def __init__(self):
		self.name = 'sigmoid'
	def forward(self, x):
		self.input = x.copy()
		self.output = 1./(1.+np.exp(-x))
		return self.output
	def backward(self):
		return self.output * (1 - self.output)


class Tanh(object):
	def __init__(self):
		self.name = 'tanh'
	def forward(self, x):
		self.input = x.copy()
		self.output =(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
		return self.output
	def backward(self):
		return 1 - self.output ** 2


class Relu(object):
	def __init__(self):
		self.name = 'relu'
	def forward(self, x):
		self.input = x.copy()
		self.output = (np.abs(x) + x)/2
		return self.output
	def backward(self):
		temp = self.input.copy()
		temp[temp>0]=1
		temp[temp<0]=0
		return temp

class Relu6(object):
	def __init__(self):
		self.name = 'relu6'
	def forward(self, x):
		self.input = x.copy()
		self.output = (np.abs(x) + x)/2
		self.output[self.output>6]=6
		return self.output
	def backward(self):
		temp = self.input.copy()
		temp[temp>6]=0
		temp[temp>0]=1
		temp[temp<0]=0
		return temp

x = np.arange(-10,10,0.1)

# activation = Sigmoid()
# activation = Tanh()
# activation = Relu()
activation = Relu6()
y = activation.forward(x).copy()
d = activation.backward()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.plot(x, y, label = "forward")
ax.plot(x, d, label = "backward")
plt.title(activation.name)
plt.legend()
plt.show()
