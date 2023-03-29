import Functions2 as func2
import numpy as np
import matplotlib.pyplot as plt

############
# Robot arm 2D, new, only one arms with one point
############

#
def parabool_path(t):
    '''A parabool in R2, which goes through (0,0), (1,1) and (2,0).
    Require: t, '''
    y = np.array([[t], [-(t-1)**2+1]]) 
    #print(y.shape)
    return y

def d_parabool_path(R, x, t):
    '''Gradient of the loss function with respect to t.'''
    d_path = np.array([[1], [-2*(t-1)]])
    Rx = np.einsum('ij, njk -> nik', R, x) 
    dt = -2*np.einsum('nji,jk  -> nik ', Rx-parabool_path(t), d_path)
    
    return dt[0,0,0]

def update_t(R, x, t, eta_t):
    '''Update the t variable using learning rate eta_t.'''
    dt = d_parabool_path(R, x, t)
    
    t = t - eta_t*dt
    print('t is: ', t)
    
    return t

def update_arm(R, x, t, eta, eta_t, U_U_trans):
    '''Update R and t for one arm'''

    gamma = parabool_path(t)
    R_new = func2.update_R(R, x, gamma, U_U_trans, eta)
    t_new = update_t(R,x,t,eta_t)
    
    return R_new, t_new

def Riemannian_grad_descent_one_arm(R_0, x, t_0, eta, eta_t, max_it):
    '''Riemannain gradient descent for one robotic arm.
    Requires: R_0, initial guess; x, initial position of the arm;
    t_0, determines the initial point on the curve; eta, learning rate for R;
    eta_t, learning rate for t; max_it, maximum number of iterations.'''
    y_it=np.zeros((max_it+1, n, 1))
    y_it[0,:,:]=func2.mat_mult_points(R_0,x)[0,:,:]
    
    U_U_trans = func2.construct_U_U_trans(n)
    R=R_0
    t=t_0
    
    for it in range(0,max_it):
        
        R, t = update_arm(R, x, t, eta, eta_t, U_U_trans)
        
        y_it[it+1,:,:] = func2.mat_mult_points(R,x)

    return R, y_it

def Riemannian_grad_descent_multiple_arms(R_0, x_0, t_0, eta, eta_t, max_it):
    '''Riemannian gradient descent for multiple robotic arms of the same length ||x_0|| (concatenated).
    Requires: R_0, initial guesses for all rotation points (dimension is n_arms x n x n);
    x_0, resting position of one arm piece; t_0, initial time guesses (dim is n_arms);
    eta, learning rate for the rotation matrices; eta_t, learning rate for the time steps;
    max_it, maximum number of iterations.'''
    
    
    
    return y_it


# main code
n=2
R=np.eye(n)
x=np.array([[[1],[0]]])
t=1
eta=1
eta_t=0.1
max_it = 10

R, y_it = Riemannian_grad_descent_one_arm(R, x, t, eta, eta_t, max_it)

#plotting results

y_to_plot = np.zeros((max_it+1, 2, n, 1))
y_to_plot[:,1,:,:] = y_it[:,:,:]
    
func2.plot_figure(y_to_plot, step=2)

num_points_gamma = 100
t_to_plot = np.linspace(0,2,num_points_gamma)
gamma_curve = parabool_path(t_to_plot)

x1 = gamma_curve[0,0,:]
x2 = gamma_curve[1,0,:]

plt.plot(x1, x2, label='desired')
plt.xlim([0,2])
plt.ylim([0,1])





# earlier
    # n_points=x.shape[0]
    # n = x.shape[1]
    # y_it=np.zeros((max_it+add_init,n_points,n,1))
    
    # if add_init:
    #     y_it[0,:,:,:]=mat_mult_points(R_0, x) #adds initial guess to y_it
    
    
    # R=R_0
    
    # #precompute U_U_trans
    # U_U_trans=func2.construct_U_U_trans(n)

    # #iteration step
    # for it in range(0,max_it):

    #     #calculating Euclidean gradient
    #     R=func2.update_R(R,x,y,U_U_trans,eta, info)
        
    #     #print('Determinant of R is ', np.linalg.det(R))
    #     #print('R after iteration ',it+1)
    #     #print(R)
    #     y_it[it+add_init,:,:,:]=func2.mat_mult_points(R,x) #the initial guess is not in this
    
    # return R, y_it


