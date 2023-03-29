import Functions2 as func2
import numpy as np
import matplotlib.pyplot as plt


##########################
#Rotation in different dimensions comparison
##########################

#iteration parameters
max_it=20
eta=1 #learning rate

#number of simulation per dimension
n_sim = 100

#dimensions
dim = np.array([2, 3, 6, 12, 20])

all_losses = np.zeros((dim.shape[0], max_it+1))
 

for k in range(0, len(dim)):
    #starting vector
    n=dim[k] # the number of dimensions
    x_0=np.zeros((1,n,1)) #the first dimension leaves room for multiple vectors
    x_0[0,0,0]=1

    #initial guess
    R_0=np.identity(n)

    for sim in range(0, n_sim):
        
        #choosing random point on the hyper sphere
        y = func2.random_point_on_n_sphere(n)
        
        #perform the algorithm using Riemannian gradient descend
        R, y_it = func2.Riem_grad_descent(R_0,x_0,y,max_it,eta)
        losses = func2.multi_loss(y_it, y)
        all_losses[k] = all_losses[k] + losses
        

all_losses = all_losses/n_sim

print(all_losses)
plt.figure()
x=np.array([k for k in range(0,max_it+1)])
for k in range(0, dim.shape[0]):
    plt.semilogy(x,all_losses[k,:])


plt.legend(dim)
plt.title('Convergence in various dimensions')
    
plt.xlabel('iteration step')
plt.ylabel('Average 2-norm')  
   
# num_it=losses.shape[0]
# x=[it for it in range(num_it)]
# if log:
#     plt.semilogy(x, losses)
#     plt.title('Loss on a log scale')
# else:
#     plt.plot(x,losses)
#     plt.title('Loss')

# plt.xlabel('iteration step')
# plt.ylabel('Average 2-norm')     
        



