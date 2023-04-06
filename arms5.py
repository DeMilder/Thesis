import Functions2 as func2
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import expm


###########################################################
# Robot arm 2D, with intermediate points
###########################################################


################################
# Define the path below
################################

def gamma(times):
    '''Calculate the coordinates of the parabool at multiple times.'''
    
    path = np.array([[[t], [-(t-1)**2+1]] for t in times]) 
    
    return path

def d_gamma(times):
    '''Calcualtes the derivative of the gamma function for multi times.'''
    
    d_gamma = np.array([[[1], [-2*(t-1)]] for t in times])
    
    return d_gamma

##############################################
# Functions used to performing the algoirhtm
##############################################

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
        x_inter_con = np.concatenate((x_inter_con, x_inter))

    x_inter_con = np.concatenate((x_inter_con, x[np.newaxis, -1]))
    
    if start_point == False:
        x_inter_con = x_inter_con[1:]
    
    
    return x_inter_con


def dt_multi(y, times, n_inter = 0):
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
        
        d_gamma = np.array([[[1], [-2*(t-1)]] for t in t_inter]) #change for a different gamma function
        front_factor = np.array([ i / (n_inter+1) for i in range(1,n_inter+2)])
        d_path_arm = np.array([front_factor[i] * d_gamma[i] for i in range(0, n_inter+1)])
        
        # print('d_path_arm is: ', d_path_arm)
        # print('y_inter is: ', y_inter)
        # print('para is: ', gamma(t_inter))
        # print('y-para is: ', y_inter-gamma(t_inter))
        
        dt_inter = -2 * np.einsum('nji, njk -> nik', y_inter-gamma(t_inter), d_path_arm)
        # print('dt_inter is: ', dt_inter)
        dt[arm_it] = (1/(n_arm*(n_inter+1))) * np.sum(dt_inter)
        
        # print('dt is: ', dt)
    return dt

def update_t(y, t, eta_t, n_inter = 0):
    '''Update the t variable using learning rate eta_t and
    the arm positions y (including the origin).'''
    dt = dt_multi(y, t, n_inter = n_inter)
    
    t = t - eta_t*dt
    
    return t

def calc_Eucl_grad_multi(y_current, t, n_inter = 0):
    '''Calculates multiple Euclidean gradients (w.r.t. R_1, R_2, ...) simultaniously.'''
    n_arm = np.shape(t)[0]
    Eucl_grad= np.zeros((n_arm, n, n))
    
    for arm_it in range((n_arm-1),-1, -1): #computationally more efficient to start at the last index
        
        #constructing intermediate time points
        if arm_it == 0:
            t_inter = intermediate_points(np.array([0,t[arm_it]]), n_inter, start_point=False)
        else:
            t_inter = intermediate_points(np.array([t[arm_it-1], t[arm_it]]), n_inter, start_point=False)
            
        #constructing intermediate y points
        y_inter = intermediate_points(np.array([y_current[arm_it], y_current[arm_it+1]]), n_inter, start_point=False)
        
        #calculating Eucl_grad
        #pre-calculations
        fact = np.array([2*i/(n_inter+1) for i in range(1,n_inter+2)])
        dist = y_inter - gamma(t_inter)
        fact_x_dist = np.array([fact[i]*dist[i] for i in range(0, n_inter+1)])
        dist_sum = np.sum(fact_x_dist, axis=0)
        
        #using pre-calculations to calculate Eucl_grad
        if arm_it == (n_arm-1):
            Eucl_grad[arm_it] = (1/(n_arm*(n_inter+1))) * np.einsum('ij, kj -> ik', dist_sum, x_0)
        else:
            Eucl_grad[arm_it] = Eucl_grad[arm_it+1] + (1/(n_arm*(n_inter+1))) * np.einsum('ij, kj -> ik', dist_sum, x_0)
        
    return Eucl_grad
    
def update_arms(R, x_0, t, eta, eta_t, U_U_trans, info=False, n_inter=0):
    '''Performs the iteration step were we update R for multiple arms.'''
    n_arm = np.shape(R)[0]
    y_current = retrieve_axis_positions(R, x_0, origin = True)
    R_update = R.copy()
    t_update = t.copy()
    
    #calculating Euclidean gradient
    Eucl_grad = calc_Eucl_grad_multi(y_current, t, n_inter = n_inter)
    
    
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
        
    y_current = retrieve_axis_positions(R_update, x_0) #speeds up convergence
        
    t_update = update_t(y_current, t_update, eta_t, n_inter = n_inter)
    
    #y_current = retrieve_axis_positions(R_update, x_0)
    
    if info:
        print('t is: ', t_update)
    
    return R_update, t_update


def identity_Rs(n, n_arms):
    '''Constructs n_arms n x n identity matrices, which can be used as R_0.'''
    R = np.zeros((n_arms, n, n))
    for it_arm in range(n_arms):
        R[it_arm,:,:] = np.eye(n)
        
    return R

def retrieve_axis_positions(R, x_0, origin = True):
    '''Calculates the positions of the rotation axes. Also gives the origin as starting point.'''
    n_arm, n = np.shape(R)[0:2]
    y = np.zeros((n_arm + 1, n, 1))
    for it_arm in range(0, n_arms):
        y[it_arm+1,:,:] = y[it_arm,:,:] + np.einsum('ij, jk -> ik', R[it_arm,:,:], x_0)
    
    if origin == False:
        y = y[1:]
        
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
    R_it = np.zeros((max_it+1, n_arms, n, n))
    R_it[0] = R_0.copy()
    t_it = np.zeros((max_it+1, n_arms))
    t_it[0,:] = t_0 #adding intial times
        
    for it in range(0,max_it):
        
        y_new = y_it[it, :, :, :].copy()
        
        print('IT NUMBER IS: ', it+1)
        
        R, t_it[it+1,:] = update_arms(R, x_0, t_it[it], eta, eta_t, U_U_trans, info=info, n_inter = n_inter)
         
    
        y_new = retrieve_axis_positions(R, x_0) #should be deleted
        
        R_it[it+1] = R
        y_it[it+1,:,:] = y_new

    return R_it, y_it, t_it


##################################
# Post-simulation analysis tools
##################################


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
            
            loss[it, arm_it+1] = np.linalg.norm(y_inter - gamma(t_inter), axis=1).mean()
        loss[it, 0] = loss[it,1:].mean()
        
    return loss

def plot_loss(losses):
    '''Plotting losses. For multiple arms, the mean value of the losses should be on the first index.
    This function does create a new figure, so use plt.figure() before hand.'''
    num_it=losses.shape[0]
    n_arms = losses.shape[1]-1 #mean value is on index 0
    
    x=[it for it in range(num_it)]
    
    if n_arms == 1:
        plt.semilogy(x, losses)
    else:
        for arm_it in range(0,n_arms+1):
            plt.semilogy(x, losses[:,arm_it])
            
    labels = ['arm piece %s' % i for i in range(0,n_arms+1)]
    labels[0] = 'mean'
    plt.legend(labels)

    plt.xlabel('iteration step')
    plt.ylabel('Average 2-norm')
    plt.title('Loss')
    
def plot_max_dist(y_it, t_it, m_inter=10000, plotting = True):
    '''This function gives (and possibly plots) the maximum distance between the arm and the gamma curve
    for every iteration step. This function does not create a figure.
    Please call plt.figure() before hand. 
    Requires: y_it, axis positions as retrieved from Riem_grad_descent_multi_arms;
    t_it, gamma variables as retrieved from Riem_grad_descent_multi_arms;
    Optional: m_inter, the number of points between every axis which are compared;
    plotting, if True it plots the max_diff_norm.
    Output: max_diff_norm, the maximum distance during every iteration step.'''
    max_it = np.shape(t_it)[0]-1
    m_inter = 1
    y_it_inter = np.zeros((max_it+1, (m_inter+1)*n_arms+1, 2, 1))
    gamma_it_inter = np.zeros((max_it+1, (m_inter+1)*n_arms+1, 2, 1))
    for it in range(0, max_it+1):
        y_it_inter[it] = intermediate_points(y_it[it], m_inter)
        t_inter = intermediate_points(np.concatenate((np.array([0]),t_it[it])), m_inter)
        gamma_it_inter[it] = gamma(t_inter)
    
    diff_norm = np.linalg.norm(y_it_inter - gamma_it_inter, axis=2)

    max_diff_norm = np.amax(diff_norm, axis=1)
    
    if plotting:
        x = [i for i in range(0, max_it+1)]
        plt.semilogy(x, max_diff_norm)
        plt.title('Max distance')
        plt.xlabel('iteration')
        plt.ylabel('|y-gamma|')
    
    return max_diff_norm

def plot_Riem_grad_R_norms(y_it, t_it, n_inter = 0, plotting = True):
    '''Çalculates the Frobenius norm of grad_R1, grad_R2, ..., grad_t 
    every iteration step.
    Requires: y_it, as given by Riem_grad_descent_multi_arms;
    t_it, as given by Riem_grad_descent_multi_arms;
    Optional: n_iter, number of intermediate points;
    plotting, is True if you want to plot the results.
    Output: all_Riem_grad_norms, contains the norm of grad_R1, grad_R2, ... for every iteration;
    all_dt_norms, contains the norm of grad_t for every iteration.'''
    n = np.shape(y_it)[2]
    max_it = np.shape(t_it)[0]-1
    n_arms = np.shape(t_it)[1]
    
    all_Eucl_grad = np.zeros((max_it+1, n_arms, n, n))
    all_Riem_grad = np.zeros((max_it+1, n_arms, n, n))
    all_dt = np.zeros((max_it+1, n_arms))
    U_U_trans = func2.construct_U_U_trans(n)
    
    for it in range(0, max_it+1):
        all_Eucl_grad[it] = calc_Eucl_grad_multi(y_it[it], t_it[it], n_inter = n_inter)
        
        all_dt[it] = dt_multi(y_it[it], t_it[it], n_inter = n_inter)
        
        for arm_it in range(0,n_arms):
            all_Riem_grad[it, arm_it] = func2.calc_Riem_grad(all_Eucl_grad[it, arm_it], R_it[it, arm_it], U_U_trans) 
    
    all_Riem_grad_norms = np.linalg.norm(all_Riem_grad, axis=(2,3))
    all_dt_norms = np.linalg.norm(all_dt, axis=1)
    
    if plotting:
        x = [i for i in range(0,max_it+1)]
        labels = ['grad_R%s' % i for i in range(1,n_arms+1)]
        labels.append('grad_t')
        
        for arm_it in range(0, n_arms):
            plt.semilogy(x, all_Riem_grad_norms[:, arm_it])

        plt.semilogy(x, all_dt_norms)
            
        plt.legend(labels)
        plt.title("Frobenius norm of the Riemannian gradient")
        plt.xlabel("iteration")
        plt.ylabel("Frobenius norm")
            
    return all_Riem_grad_norms, all_dt_norms

def plot_parabool_path(start, stop, n_points):
    '''Plot the parabool paht starting at t=start and ending at t=stop.
    n_points number of points are plotted.'''
    num_points_gamma = 100
    t_to_plot = np.linspace(0,2,num_points_gamma)
    gamma_curve = gamma(t_to_plot)
    
    x1 = gamma_curve[:,0,0]
    x2 = gamma_curve[:,1,0]
    
    plt.plot(x1, x2, label='desired')
    plt.xlim([0,2])
    plt.ylim([0,2])
    plt.legend()
    plt.title('robotic arm')
      
    
    

################################
# main code below
################################
    

# Setting simulation variables
n=2 #dimension, can not be changed in this version yet.
n_arms = 3
R_0 = identity_Rs(n, n_arms)
#theta = 1/4 * np.pi
#R_0[0] = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
t_0 = np.linspace(0,1.5,n_arms+1)[1:]    
x_0=np.zeros((n,1))
x_0[0,0]=0.8
x_0_length = np.linalg.norm(x_0)
t_0 = np.linspace(0.5*x_0_length, 0.5*n_arms*x_0_length, n_arms) 

#eta=np.array([0.5, 0.5, 0.5, 0.5, 0.5])
eta=2
eta_t=1
max_it = 200
n_inter = 9
n_to_plot=5

# printing some information
y=retrieve_axis_positions(R_0, x_0)
print('n_arms is: ', n_arms)
print('initial coordinate are ', y)
print('initial t is: ', t_0)


#performing the algorithm
start_time = time.time()
R_it, y_it, t_it = Riemannian_grad_descent_multi_arms(R_0, x_0, t_0, eta, eta_t, max_it, info=False, n_inter = n_inter)
execution_time = time.time()-start_time
print('The algorithm ran for: ', execution_time, ' seconds')

#analyses of the results
#calculating losses
loss = calc_loss(y_it, t_it)

#plotting results 
#calculating Riem_grad_norms

# plotting arm
plt.close('all')
plt.figure()
func2.plot_figure(y_it, step=max_it/n_to_plot)

# plotting gamma curve
plot_parabool_path(0, 2, 200)

#plotting the point on the curve on which we are fitting at the end
x_scatter = gamma(t_it[max_it])[:,0]
y_scatter = gamma(t_it[max_it])[:,1]
    
plt.scatter(x_scatter, y_scatter)

#plotting loss functions
plt.figure()
plot_loss(loss)

#plotting max_dist
plt.figure()
plot_max_dist(y_it, t_it)

#plotting Riem grad w.r.t. to R
plt.figure()
all_Riplot_Riem_grad_R_norms, all_dt_norms = plot_Riem_grad_R_norms(y_it, t_it, n_inter=n_inter)





