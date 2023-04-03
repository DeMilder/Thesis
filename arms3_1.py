import Functions2 as func2
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

############
# Robot arm 2D, same as arms3, however now the gradient of the loss function is correct.
# As a result the arm often gets stuck in a local minimum.
############

#

def parabool_path(t):
    '''Gives one point on the path at time t.'''
    path = np.array([[t], [-(t-1)**2+1]]) 
    
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

def update_arms(R, x_0, t, eta, eta_t, U_U_trans, info=False):
    '''Update R for one arm'''
    n_arm = np.shape(R)[0]
    n = np.shape(R)[1]
    Eucl_grad= np.zeros((n_arm, n, n))
    y_current = retrieve_axis_positions(R, x_0, origin = True)
    R_update = R.copy()
    t_update = t.copy()
    
    
    for arm_it in range(0, n_arm):
        #if arm_it == (n_arm-1): 
            #print("test1: ", y_current[arm_it] - parabool_path(t[arm_it]))
            #print("test2: ", parabool_path(t[arm_it]))
            #print("test3: ", x_0)
            
        ######
        #this is code based on arm3
        ########
        # y_star = parabool_path(t[arm_it]) - y_current[arm_it]
        # print('y_star is:', y_star, ' for arm piece ', arm_it)
        # R_update[arm_it,:, :] = func2.update_R(R_update[arm_it], x_0[np.newaxis,:,:], y_star, U_U_trans, eta[arm_it])
        
        # y_current = retrieve_axis_positions(R, x_0)
        #######
        #end
        ######
        for arm_it_it in range((n_arm-1), (arm_it-1), -1):
            Eucl_grad[arm_it] = Eucl_grad[arm_it] + 2*np.einsum('ij, kj -> ik', y_current[arm_it_it] - parabool_path(t[arm_it_it]), x_0)
        
        in_exp=func2.calc_Riem_grad(Eucl_grad[arm_it],R[arm_it],U_U_trans)
        R_exp = expm(-eta[arm_it]*in_exp)
        R_update[arm_it] = np.einsum('ij, jk-> ik', R[arm_it], R_exp)
        
    y_current = retrieve_axis_positions(R_update, x_0, origin = False)
        
        
    #     if info:
    #         print('Current arm_it is: ', arm_it)
    #         print("The new R is: ", R_update[arm_it], 'for arm piece ', arm_it)
    #         print("The determinant is: ", np.linalg.det(R_update[arm_it]))
    #         print("Should be idenity: ", np.einsum("ij,kj -> ik", R_update[arm_it], R_update[arm_it]))
            
    t_update = update_t(y_current[1:,:,:], t_update, eta_t)
    
    return R_update, t_update


def identity_Rs(n, n_arms):
    R = np.zeros((n_arms, n, n))
    for it_arm in range(n_arms):
        R[it_arm,:,:] = np.eye(n)
        
    return R

def retrieve_axis_positions(R, x_0, origin = True):
    '''Calculates the positions of the rotation axes. Also gives the origin as starting point.'''
    n_arm, n = np.shape(R)[0:2]
    y = np.zeros((n_arm+origin, n, 1))
    for it_arm in range(0, n_arms):
        y[it_arm+origin,:,:] = y[it_arm,:,:] + np.einsum('ij, jk -> ik', R[it_arm,:,:], x_0)
        
    return y

def Riemannian_grad_descent_multi_arms(R_0, x_0, t_0, eta, eta_t, max_it):
    '''Riemannian gradient descent for multiple robotic arms of the same length ||x_0|| (concatenated).
    Requires: R_0, initial guesses for all rotation points (dimension is n_arms x n x n);
    x_0, resting position of one arm piece; t_0, initial time guesses (dim is n_arms);
    eta, learning rate for the rotation matrices; eta_t, learning rate for the time steps;
    max_it, maximum number of iterations.
    Output: R, rotation of every arm piece w.r.t. the x-axis;
    y_it, an (n_it+1) x (n_arms+1) array containing the coordinates of the rotation axes
    (including the origin) for every iteration (including the initial position);
    t_it, t values for the path for every iteration (including the initial t values).'''
    n = np.shape(R_0)[1]
    n_arms = np.shape(R_0)[0]
    
    x = np.zeros((n_arms, 2, 1))
    x[0] = np.einsum('ij, jk -> ik', R_0[0,:,:], x_0)
    
    x = retrieve_axis_positions(R_0, x_0) #initial positions
    
    y_it=np.zeros((max_it+1, n_arms+1, n, 1))
    y_it[0,:,:,:] = x #save intial positions
    
    U_U_trans = func2.construct_U_U_trans(n)
    R=R_0.copy()
    R_new=R.copy()
    t_it = np.zeros((max_it+1, n_arms))
    t_it[0,:] = t_0 #adding intial times
    
    #x_0_extra_dim = x_0[np.newaxis,:,:] #we need a 3d array for update arm.
    
    for it in range(0,max_it):
        
        y_new = y_it[it, :, :, :].copy()
        print('IT NUMBER IS: ', it)
        
        #print('t_it is: ', t_it[it])
        
        R_new, t_it[it+1,:] = update_arms(R, x_0, t_it[it], eta, eta_t, U_U_trans, info=True)
         
        
        #t_it[it+1,:] = update_t(y_new[1:n_arms+1,:,:],t_it[it,:],eta_t) #not including the origin
            
        R = R_new.copy()
        y_new = retrieve_axis_positions(R, x_0) #should be deleted
        
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
n_arms = 2
R_0 = identity_Rs(n, n_arms)
theta = 1/4 * np.pi
#R_0[0] = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
t_0 = np.linspace(0.5,1.5,n_arms)    
x_0=np.zeros((n,1))
x_0[0,0]=1
x_0_length = np.linalg.norm(x_0)
t_0 = np.linspace(0.5*x_0_length, 0.5*n_arms*x_0_length, n_arms) 


y=retrieve_axis_positions(R_0, x_0)
print('initial coordinate are ', y)
print('initial t is: ', t_0)

eta=np.array([0.5, 0.5, 1, 1, 1])
eta_t=0.001
max_it = 200
n_to_plot=5

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
func2.plot_figure(y_it, step=max_it/n_to_plot)

num_points_gamma = 100
t_to_plot = np.linspace(0,2,num_points_gamma)
gamma_curve = parabool_path_multi(t_to_plot)

x1 = gamma_curve[:,0,0]
x2 = gamma_curve[:,1,0]

plt.plot(x1, x2, label='desired')
plt.xlim([0,2])
plt.ylim([0,2])
plt.legend()
plt.title('robotic arm')

#plotting the point on the curve on which we are fitting at the end
x_scatter = np.zeros(n_arms)
y_scatter = np.zeros(n_arms)
for k in range(0, n_arms):
    x_scatter[k] = parabool_path(t_it[max_it,k])[0,0]
    y_scatter[k] = parabool_path(t_it[max_it,k])[1,0]

    
plt.scatter(x_scatter, y_scatter)

#plotting loss functions
plt.figure()
labels = ['' for i in range(0, n_arms+1)]

func2.plot_loss(loss[:,0])
labels[0] = 'mean'

for arm in range(1, (n_arms+1)):
    func2.plot_loss(loss[:,arm])
    labels[arm] = 'arm %s' % arm

plt.legend(labels)
plt.title('loss of robotic arm')





