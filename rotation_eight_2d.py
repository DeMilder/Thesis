import Functions2 as func2
import numpy as np
import matplotlib.pyplot as plt
#scipy could also be useful

#############################
#Rotation of the eight figure
#############################

#iteration parameters
max_it=100
eta=0.5 #learning rate

#given is the eight figure
n_points=200
x=func2.eight(n_points)
angle=0.5*np.pi

#simpel test case
#x=np.array([[[0],[0]],[[1],[0]]]) test case
#angle=0.5*np.pi

#the disired solutions
R_real=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle), np.cos(angle)]])
y=func2.mat_mult_points(R_real, x)

#initial guess
R_0=np.identity(2)

#perform the algorithm using Riemannian gradient descend
R, y_it = func2.Riem_grad_descent(R_0,x,y,max_it,eta, info=True)

print('y_it_shape is: ', y_it.shape)
    
#plots iteration figures
plt.close('all') #close old figure windows (if there are any)
plt.figure()
func2.plot_figure(y_it,y,step=5)

#plot loss function
plt.figure()
losses=func2.multi_loss(y_it,y)
func2.plot_loss(losses)





