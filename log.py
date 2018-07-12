import numpy as np
#import panda as pd
from sklearn import datasets

class log:
	def __init__(self, lr=0.01 ,it=10000,fit_int=True, verbose=False):
		self.lr=lr
		self.it=it
		self.fit_int=fit_int


	def add_int(self,d):
		x=np.asarray(d)
		inter=np.ones((x.shape[0],1))
		print(np.concatenate((inter, x), axis=1))
		
		return np.concatenate((inter, x), axis=1)

	def sig(self, x):
		return 1/(1+ np.exp(-x))

	def loss(self , h ,y):
		s= ((-y*np.log(h)) - (1 - y) * np.log(1 - h))
		print("hello")
		print(s)
		print("vscd")
		return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
		#return ((-y*np.log(h)) - (1 - y) * np.log(1 - h)).mean()

		

	def fit(self, x, y):
		if self.fit_int:
			x= self.add_int(x)
			print(x)
		self.theta = np.zeros(x.shape[1])

		for i in range(self.it):
			z=np.dot(x,self.theta)
			h=self.sig(z)
			grad= np.dot(x.T, (h-y)) / y.size
			self.theta -= self.lr * grad

		z=np.dot(x,self.theta)
		h=self.sig(h)
		print(self.loss(h, y))

	def pred(self,x):
		q=0
		k=0
		j=np.asarray(x);
	
		print(np.dot(x,self.theta))			
		
		return self.sig(np.dot(x,self.theta))


	def get(self,x,t):
		print((self.pred(x)))
		return bool(self.pred(x)>=t)

def run():
	
	iris = datasets.load_iris()
	x = iris.data[:,:2]
	y = (iris.target!=0)*1
	

	print(x)
	print(y)
	a = log(lr=0.1, it = 10000)
	a.fit(x,y)
	#print(a.loss(x,y))
	q=[1,5.1,3.5]
	t=0.5
	print("vds;kjbeqk")
	print(a.get(q,t))




if __name__ == '__main__':
	run()