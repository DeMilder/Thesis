import Functions2 as func2
import numpy as np
import matplotlib.pyplot as plt
import time

##########################
#Rotation of a vector n-dimensions
##########################

#iteration parameters
max_it=20
eta=1 #learning rate

#dimensions
n=10

#starting vector
x=np.zeros((1,n,1)) #the first dimension leaves room for multiple vectors
x[0,0,0]=1

#the disired solutions
#y = func2.random_point_on_n_sphere(n) %random point
y = np.zeros((1,n,1))
y[0, -1 ,0] = 0.5*np.sqrt(2)
y[0, -2, 0] = 0.5*np.sqrt(2)

#initial guess
R_0=np.identity(n)

start_time = time.time()
#perform the algorithm using Riemannian gradient descend
R, y_it = func2.Riem_grad_descent(R_0,x,y,max_it,eta, info=False)

execution_time = time.time()-start_time

print("The code ran for: %s seconds" % execution_time)

is_id= np.einsum('ij,kj->ik',R,R)
    

#plot loss function
plt.figure()
losses=func2.multi_loss(y_it,y)
func2.plot_loss(losses)

