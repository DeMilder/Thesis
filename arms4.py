import Functions2 as func2
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

############
# Robot arm 2D, with intermediate points, work in progress
############

#
def intermediate_points(x, n_inter, start_point = True):
    '''Give a np.array x (of any dimension). This function adds n_inter intermediate
    points between every point on axis 0.
    Requires: x, np.array; n_inter, number of intermediate point
    Output: x_inter_con, x array with intermediate points.
    '''
    n_x = np.shape(x)[0]
    x_inter_con = np.linspace(x[0],x[1], n_inter+1, endpoint = False, axis=0)
    
    for x_it in range(1,n_x-1):
        x_inter = np.linspace(x[x_it],x[x_it+1], n_inter+1, endpoint = False, axis=0) #start and end
        x_inter_con = np.append(x_inter_con, x_inter)
        
    x_inter_con = np.concatenate((x_inter_con, x[np.newaxis, -1]))
    
    if start_point == False:
        x_inter_con = x_inter_con[1:]
    
    
    return x_inter_con

def parabool_path(t):
    '''Gives one point on the path at time t.'''
    
    path = np.array([[t], [-(t-1)**2+1]]) 
    
    return path

def parabool_path_multi(times):
    '''Calculate the coordinates of the parabool at multiple times.'''
    
    path = np.array([[[t], [-(t-1)**2+1]] for t in times]) 
    
    return path


def d_parabool_path_multi(y, times, n_inter = 0):
    '''Calculate the gradient w.r.t. to t for multple times using the arm
    positions y (including the origin).'''
    n_arm = np.shape(times)[0]
    dt = np.zeros(n_arm)
    # print('times is: ', times)
    for arm_it in range(0, n_arm):
        if arm_it ==0:
            # print('tets is: ', np.array([0,times[arm_it]]))
            t_inter = intermediate_points(np.array([0,times[arm_it]]), n_inter, start_point = False)
        else:
            t_inter = intermediate_points(np.array([times[arm_it-1], times[arm_it]]), n_inter, start_point = False)
            
        y_inter = intermediate_points(np.array([y[arm_it], y[arm_it+1]]), n_inter, start_point = False)
        #print('y_inter is: ', y_inter)
        # print('t_inter is: ', t_inter)
        
        d_gamma = np.array([[[1], [-2*(t-1)]] for t in t_inter])
        front_factor = np.array([ i / (n_inter+1) for i in range(1,n_inter+2)])
        d_path_arm = np.array([front_factor[i] * d_gamma[i] for i in range(0, n_inter+1)])
        
        # print('d_path_arm is: ', d_path_arm)
        # print('y_inter is: ', y_inter)
        # print('para is: ', parabool_path_multi(t_inter))
        # print('y-para is: ', y_inter-parabool_path_multi(t_inter))
        
        dt_inter = -2 * np.einsum('nji, njk -> nik', y_inter-parabool_path_multi(t_inter), d_path_arm)
        # print('dt_inter is: ', dt_inter)
        dt[arm_it] = (1/(n_arm*(n_inter+1))) * np.sum(dt_inter)
        
        print('dt is: ', dt)
    return dt

def update_t(y, t, eta_t, n_inter = 0):
    '''Update the t variable using learning rate eta_t and
    the arm positions y (including the origin).'''
    dt = d_parabool_path_multi(y, t, n_inter = n_inter)
    
    t = t - eta_t*dt
    
    return t

def update_arms(R, x_0, t, eta, eta_t, U_U_trans, info=False, n_inter=0):
    '''Update R for one arm'''
    n_arm = np.shape(R)[0]
    n = np.shape(R)[1]
    Eucl_grad= np.zeros((n_arm, n, n))
    y_current = retrieve_axis_positions(R, x_0, origin = True)
    R_update = R.copy()
    t_update = t.copy()
    
    #calculating Euclidean gradient
    
    for arm_it in range((n_arm-1),-1, -1):
        if arm_it == 0:
            t_inter = intermediate_points(np.array([0,t[arm_it]]), n_inter, start_point=False)
        else:
            t_inter = intermediate_points(np.array([t[arm_it-1], t[arm_it]]), n_inter, start_point=False)
            
        #print('t_inter is: ', t_inter)
        
        y_inter = intermediate_points(np.array([y_current[arm_it], y_current[arm_it+1]]), n_inter, start_point=False)
        fact = np.array([2*i/(n_inter+1) for i in range(1,n_inter+2)])
        dist = y_inter - parabool_path_multi(t_inter)
        fact_x_dist = np.array([fact[i]*dist[i] for i in range(0, n_inter+1)])
        
        print('fact is: ', fact)
        print('dist is: ', dist)
        print('fact_x_dist is: ', fact_x_dist)
        
        dist_sum = np.sum(fact_x_dist, axis=0)
        print('dist_sum is: ', dist_sum)
        
        if arm_it == (n_arm-1):
            Eucl_grad[arm_it] = (1/(n_arm*(n_inter+1))) * np.einsum('ij, kj -> ik', dist_sum, x_0)
        else:
            Eucl_grad[arm_it] = Eucl_grad[arm_it+1] + (1/(n_arm*(n_inter+1))) * np.einsum('ij, kj -> ik', dist_sum, x_0)
    
    
    #Calculating Riem_grad and updating the R's
    for arm_it in range(0, n_arm):
        
        in_exp=func2.calc_Riem_grad(Eucl_grad[arm_it],R[arm_it],U_U_trans)
        
        if np.size(eta) > 1:
            R_exp = expm(-eta[arm_it]*in_exp)
        else:
            R_exp = expm(-eta*in_exp)
            
        R_update[arm_it] = np.einsum('ij, jk-> ik', R[arm_it], R_exp)
        
        if info:
            print('Current arm_it is: ', arm_it)
            print("The new R is: ", R_update[arm_it], 'for arm piece ', arm_it)
            print("The determinant is: ", np.linalg.det(R_update[arm_it]))
            print("Should be idenity: ", np.einsum("ij,kj -> ik", R_update[arm_it], R_update[arm_it]))
        
    y_current = retrieve_axis_positions(R_update, x_0)
            
    t_update = update_t(y_current, t_update, eta_t, n_inter = n_inter)
    
    if info:
        print('t is: ', t_update)
    
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

def Riemannian_grad_descent_multi_arms(R_0, x_0, t_0, eta, eta_t, max_it, info = False, n_inter = 0):
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
        
    for it in range(0,max_it):
        
        y_new = y_it[it, :, :, :].copy()
        print('IT NUMBER IS: ', it+1)
        
        R_new, t_it[it+1,:] = update_arms(R, x_0, t_it[it], eta, eta_t, U_U_trans, info=info, n_inter = n_inter)
         
        R = R_new.copy()
        y_new = retrieve_axis_positions(R, x_0) #should be deleted
        
        y_it[it+1,:,:] = y_new

    return R, y_it, t_it

def calc_loss(y_it, t_it, n_inter = 0):
    '''Calculates the loss function.
    Requires: y_it, robot arm positions of multiple iterations; 
    t_it, time update of multiple iterations.
    Output: loss, an n_it x (n_arms+1) array containing in the first column the loss
    averaged over all the arms and in the following columns the loss w.r.t. every arm.
    '''
    n_it_p_init = np.shape(t_it)[0]
    n_arms = np.shape(t_it)[1]
    loss = np.zeros((n_it_p_init,n_arms+1))
    for it in range (0, n_it_p_init):
        for arm_it in range(0,n_arms):
            y_inter = intermediate_points(np.array([y_it[it, arm_it], y_it[it, arm_it+1]]), n_inter, start_point=False)
            
            if arm_it == 0: 
                t_inter = intermediate_points(np.array([0, t_it[it, arm_it]]), n_inter, start_point=False)
            else:
                t_inter = intermediate_points(np.array([t_it[it,arm_it-1], t_it[it, arm_it]]), n_inter, start_point=False)
            
            loss[it, arm_it+1] = np.linalg.norm(y_inter - parabool_path_multi(t_inter), axis=1).mean()
        loss[it, 0] = loss[it,1:].mean()
        
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

#eta=np.array([0.5, 0.5, 0.5, 0.5, 0.5])
eta=2
eta_t=0.5
max_it = 50
n_inter = 9
n_to_plot=5


print('test done')

#performing the algorithm
R, y_it, t_it = Riemannian_grad_descent_multi_arms(R_0, x_0, t_0, eta, eta_t, max_it, info=False, n_inter = n_inter)

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





