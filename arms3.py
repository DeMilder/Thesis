import Functions2 as func2
import numpy as np
import matplotlib.pyplot as plt

############
# Robot arm 2D, new, multiple arms each one end point
############

#

def parabool_path(t):
    '''Gives one point on the path at time t.'''
    path = np.array([[[t], [-(t-1)**2+1]]]) 
    
    return path

def parabool_path_multi(times):
    '''Calculate the coordinates of the parabool at multiple times.'''
    n_t = np.shape(times)[0]
    path = np.zeros((n_t,2,1))
    for it_t in range(n_t):
        path[it_t,:,:] = np.array([[times[it_t]], [-(times[it_t]-1)**2+1]]) 
        
    return path

def d_parabool_path_multi(y, t):
    '''Calculate the gradient w.r.t. to t for multple times using the arm
    positions y (not including the origin).'''
    n_arms = np.shape(t)[0]
    d_path = np.zeros((n_arms, 2, 1))
    
    for it_arm in range(n_arms):
        d_path[it_arm,:,:] = np.array([[1], [-2*(t[it_arm]-1)]])  
    #print('arm posisitons: ', y)
    #print('parabool path positions: ', parabool_path_multi(t))
    dt = -2*np.einsum('nji, njk -> nik', (y - parabool_path_multi(t)), d_path)
    #print('dt: ', dt)
    return np.squeeze(dt) #removes redundant dimensions

def update_t(y, t, eta_t):
    '''Update the t variable using learning rate eta_t and
    the arm positions y (not including the origin).'''
    dt = d_parabool_path_multi(y, t)
    
    t = t - eta_t*dt
    print('t is: ', t)
    
    return t

def update_arm(R, x, y_star, eta, eta_t, U_U_trans):
    '''Update R and t for one arm'''
    R_new = func2.update_R(R, x, y_star, U_U_trans, eta)
    # print('update from update arm: ', R_new)
    return R_new


def Riemannian_grad_descent_one_arm(R_0, x, t_0, eta, eta_t, max_it):
    '''Riemannain gradient descent for one robotic arm.
    Requires: R_0, initial guess; x, initial position of the arm;
    t_0, determines the initial point on the curve; eta, learning rate for R;
    eta_t, learning rate for t; max_it, maximum number of iterations.'''
    n = np.size(R_0)[1] #dimension
    
    y_it=np.zeros((max_it+1, n, 1))
    y_it[0,:,:]=func2.mat_mult_points(R_0,x)[0,:,:]
    
    U_U_trans = func2.construct_U_U_trans(n)
    R=R_0.copy()
    t=t_0.copy()
    
    for it in range(0,max_it):
        
        R, t = update_arm(R, x, t, eta, eta_t, U_U_trans)
        
        y_it[it+1,:,:] = func2.mat_mult_points(R,x)

    return R, y_it

def identity_Rs(n, n_arms):
    R = np.zeros((n_arms, n, n))
    for it_arm in range(n_arms):
        R[it_arm,:,:] = np.eye(n)
        
    return R

def retrieve_axis_possitions(R, x_0):
    '''Calculates the positions of the rotation axes. Also gives the origin as starting point.'''
    n_arm, n = np.shape(R)[0:2]
    y = np.zeros((n_arm+1, n, 1))
    for it_arm in range(0, n_arms):
        y[it_arm+1,:,:] = y[it_arm,:,:] + np.einsum('ij, jk -> ik', R[it_arm,:,:], x_0)
        
    return y

def Riemannian_grad_descent_multi_arms(R_0, x_0, t_0, eta, eta_t, max_it):
    '''Riemannian gradient descent for multiple robotic arms of the same length ||x_0|| (concatenated).
    Requires: R_0, initial guesses for all rotation points (dimension is n_arms x n x n);
    x_0, resting position of one arm piece; t_0, initial time guesses (dim is n_arms);
    eta, learning rate for the rotation matrices; eta_t, learning rate for the time steps;
    max_it, maximum number of iterations.'''
    n = np.shape(R_0)[1]
    n_arms = np.shape(R_0)[0]
    
    x = np.zeros((n_arms, 2, 1))
    x[0] = np.einsum('ij, jk -> ik', R_0[0,:,:], x_0)
    
    x = retrieve_axis_possitions(R_0, x_0) #initial positions
    
    y_it=np.zeros((max_it+1, n_arms+1, n, 1))
    y_it[0,:,:,:] = x #save intial positions
    
    U_U_trans = func2.construct_U_U_trans(n)
    R=R_0.copy()
    R_new=R.copy()
    t_it = np.zeros((max_it+1, n_arms))
    t_it[0,:] = t_0
    
    x_0_extra_dim = x_0[np.newaxis,:,:] #we need a 3d array for update arm.
    
    for it in range(0,max_it):
        
        y_new = y_it[it, :, :, :].copy()
        print('IT NUMBER IS: ', it)
        
        for it_arm in range(0, n_arms):
            
            
            y_star = parabool_path(t_it[it, it_arm])-y_new[it_arm, :, :] #gamma - previous rotation point
            #print('arm piece is ', it_arm, ', y_star is: ', y_star)
            #print('y_star size: ', np.shape(y_star))
            R_new[it_arm, :, :] = update_arm(R[it_arm,:,:], x_0_extra_dim, y_star, eta, eta_t, U_U_trans)
            
            #compensating for rotations
            # for it_next_arms in range(it_arm+1, n_arms):
            #     R_new[it_next_arms,:,:] = np.einsum('ij, jk -> ik', R_new[it_arm, :, :], R_new[it_next_arms,:,:] ) 
            
            y_new = retrieve_axis_possitions(R_new, x_0) #retrieve_axis_positions also outputs the origin
         
        
        t_it[it+1,:] = update_t(y_new[1:n_arms+1,:,:],t_it[it],eta_t) #not including the origin
            
        R = R_new.copy()
            
        y_it[it+1,:,:] = y_new

    return R, y_it, t_it

def calc_loss(y_it, t_it):
    '''Calculates the loss function.
    Requires: y_it, robot arm positions of multiple iterations; 
    t_it, time update of multiple iterations.
    Output: loss, an n_it x (n_arms+1) array containing in the first column the loss
    averaged over all the arms and in the following columns the loss w.r.t. every arm.
    '''
    n_it_p_init = np.shape(t_it)[0]
    n_arms = np.shape(t_it)[1]
    loss = np.zeros((n_it_p_init,n_arms+1))
    
    for it in range(0,n_it_p_init):
        loss_all_arms = np.squeeze(np.linalg.norm(y_it[it,1:n_arms+1,:,:] - parabool_path_multi(t_it[it]), axis=1))
        #print('loss_all_norms is: ', loss_all_arms)
        loss[it,0]=loss_all_arms.mean()
        loss[it,1:(n_arms+1)] = loss_all_arms
        
    return loss
    
    
n=2
n_arms = 5
R_0 = identity_Rs(n, n_arms)
theta = 1/4 * np.pi
#R_0[0] = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
t_0 = np.linspace(0.5,1.5,n_arms)    
x_0=np.zeros((n,1))
x_0[0,0]=0.5

y=retrieve_axis_possitions(R_0, x_0)
print('initial coordinate are ', y)
print('initial t is: ', t_0)

eta=0.3
eta_t=0.1
max_it = 50

#path = parabool_path_multi(t_0)
#dt = d_parabool_path_multi(y[1:n_arms+1], t_0)
#('test done')

#performing the algorithm
R, y_it, t_it = Riemannian_grad_descent_multi_arms(R_0, x_0, t_0, eta, eta_t, max_it)

#calculating losses
loss = calc_loss(y_it, t_it)

#plotting results  
# plotting arm
plt.close('all')
plt.figure()
func2.plot_figure(y_it, step=5)

num_points_gamma = 100
t_to_plot = np.linspace(0,2,num_points_gamma)
gamma_curve = parabool_path_multi(t_to_plot)

x1 = gamma_curve[:,0,0]
x2 = gamma_curve[:,1,0]

plt.plot(x1, x2, label='desired')
plt.xlim([0,2])
plt.ylim([0,2])
plt.legend()

#plotting loss functions
plt.figure()
labels = ['' for i in range(0, n_arms+1)]

func2.plot_loss(loss[:,0])
labels[0] = 'mean'

for arm in range(1, (n_arms+1)):
    func2.plot_loss(loss[:,arm])
    labels[arm] = 'arm %s' % arm

plt.legend(labels)





