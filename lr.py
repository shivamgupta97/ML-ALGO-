from numpy import *


def step(b,m,points,lr):
	bg=0
	mg=0
	n=float(len(points))
	for i in range(0,len(points)):
		x=points[i,0]
		y=points[i,1]
		bg += -(2/n) * (y - (m * x) + b)
		mg += -(2/n) * (x) *(y - (m * x) + b)
		nb=b- (lr * bg)
		nm= m - (lr * mg)
	return [nb,nm]




def grad(points,im,ib,lr,it):
	m=im
	b=ib
	for i in range(it):
		b,m=step(b,m,array(points),lr)
	return [b,m]




def run():
	points = genfromtxt("data.csv", delimiter=",")
	lr=0.0001
	ib=0
	im=0
	it=10000
	#y=mx+b
	print("start")
	[b,m] = grad(points, im,ib,lr,it)
	print(b,m)

if __name__ == '__main__':
	run()