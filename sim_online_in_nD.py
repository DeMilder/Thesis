import Functions2 as func2
import numpy as np
import matplotlib.pyplot as plt

##########################
#Rotation of a vector in n-dimensions with multiple starting positions
##########################

#iteration parameters
max_it=20 #iterations for each simulation
eta=1 #learning rate

n_sim = 5 #total iterations = n_sim*max_it

#dimensions
n=12
    
y_multi_sim_it = np.zeros((n_sim*max_it+1, 1, n, 1))
losses = np.zeros(n_sim*max_it+1)

#starting vector
x_0=np.zeros((1,n,1)) #the first dimension leaves room for multiple vectors
x_0[0,0,0]=1

#initial guess
R_0=np.identity(n)


for sim in range(0, n_sim):
    
    #choosing random point on the hyper sphere
    y = func2.random_point_on_n_sphere(n)
    
    if sim==0:
    #perform the algorithm using Riemannian gradient descend
        R, y_it = func2.Riem_grad_descent(R_0,x_0,y,max_it,eta)
        
        y_multi_sim_it[0: (max_it+1) ,:,:,:] = y_it #contains also the initial position
        losses[0:(max_it+1)] = func2.multi_loss(y_it, y)
    else:
        R_0 = R
        
        R, y_it = func2.Riem_grad_descent(R_0,x_0,y,max_it,eta, add_init=False) 
        #add_init=False, since the initial possition is equal to the final position of the previous simulation
        y_multi_sim_it[(max_it*sim+1): (max_it*(sim+1)+1),:,:,:] = y_it
        losses[(max_it*sim+1): (max_it*(sim+1)+1)] = func2.multi_loss(y_it, y)
    

#plot loss function
plt.figure()
func2.plot_loss(losses)
