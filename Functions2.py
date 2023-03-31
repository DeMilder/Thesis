import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.linalg import expm

import Functions2 as func2


def eight(num_points=100):
    '''Para metric function that creates a 8-figure in 2D .
    It outputs num_points many points in a tensor, where the coordinates are storred column-wise. 
    The first dimension is the point index.
    The second is are the rows of the point vector (is 2).
    The third is are the columns of the point vector (is 3)'''
    x=np.zeros((num_points,2,1))
    x_1=0.5*np.sin(2*np.linspace(0,2*math.pi,num_points))
    x_2=np.sin(np.linspace(0,2*math.pi,num_points)) 
        
    x[:,0,0]=x_1
    x[:,1,0]=x_2
    return x

def random_point_on_n_sphere(n):
    '''Selects a random point on a n dimensional hypersphere'''
    x = np.random.normal(size=(1,n,1))
    x = x / np.linalg.norm(x)
    return x

def plot_figure(x,y=None, step=1, info = False):
    '''Given a four dimensional array x, this function plots the figure.
    First dimension of x contains all the pictures
    Second dimensions of x are the datapoints per picture
    Third dimension is the row of the data (x and y)
    Fourth dimension is equal to the columns of the data (always 1)'''
    if info:
        print('given data to plot is ', x)
    #clear figure
    num_pictures=x.shape[0]
    num_points_per_picture=x.shape[1]
    plot_data=np.zeros((2, num_points_per_picture))
    label_format='it {it_step:.0f}'
    for picture in range(num_pictures):
        if picture%step == 0:
            plot_data[:,:] = np.array([x[picture,:,0,0],x[picture,:,1,0]])
            # print('plot_data is: ', plot_data)
            plt.plot(plot_data[0,:],plot_data[1,:],label=label_format.format(it_step=picture))
        
    #print('plot_data is', plot_data)
    
    #plotting y
    if y is not None:
        plot_data[:,:]=np.array([y[:,0,0],y[:,1,0]])
        plt.plot(plot_data[0,:],plot_data[1,:],label='desired')
    
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.show()
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    
    
def plot_loss(losses, log=True):
    num_it=losses.shape[0]
    x=[it for it in range(num_it)]
    if log:
        plt.semilogy(x, losses)
        plt.title('Loss on a log scale')
    else:
        plt.plot(x,losses)
        plt.title('Loss')

    plt.xlabel('iteration step')
    plt.ylabel('Average 2-norm')

def mat_mult_points(R,x):
    '''Matrix multiplication of all the points in x by R.'''
    return np.einsum('ij,njk->nik', R, x)

def mat_mult_basis(R,U):
    return np.einsum('ij,njk->nik',R,U)

def loss(R,x,y):
    '''Loss function which is equal to the average 2-norm of Rx-y.'''
    all_norms=np.linalg.norm(mat_mult_points(R,x)-y, axis=1)
    #print('all_norms is ',all_norms)
    return all_norms.mean()

def multi_loss(y_it,y):
    np.linalg.norm(y_it-y)
    num_it=np.shape(y_it)[0]
    losses=np.zeros(num_it)
    for it in range(num_it):
        all_norms=np.linalg.norm(y_it[it]-y,axis=1)
        losses[it]=all_norms.mean()
    
    return losses
    

def d_loss(R,x,y):
    '''Differential of the loss function.'''
    num_points=x.shape[0]
    R_x_min_y=np.einsum('ij,njz->niz',R,x) - y
    d_l=(2/num_points)*np.einsum('niz,njz->ij',R_x_min_y,x)
    return d_l

def constr_skew_basis(dim=2):
    '''Constructs a basis of skew-symmetric matrices.
    Optional: dim, dimension of the embedding space of SO(n) i.e. n.
    Output: basis, the skew-symmetric basis S_{skew}(dim).'''
    num_basis=int(dim*(dim-1)/2)
    basis=np.zeros((num_basis,dim,dim))
    basis_it=0
    for i in range(0,dim-1,1):
        for j in range(i+1,dim,1):
            basis[basis_it,i,j]=1/2*np.sqrt(2)
            basis[basis_it,j,i]=-1/2*np.sqrt(2)
            basis_it=basis_it+1
   
    return basis



def calc_Riem_grad(Eucl_grad,R,U_U_trans):
    '''Calculating the Riemannian gradient by projecting the Eucl_grad onto the tangent space T_R M.
    Requires: Eucl_grad, the Euclidean gradient; R, the current iteration matrix;
    U_U_trans, the precomputed UU^T where U is a matrix contining the skew-symmetric basis vectors.
    Output: in_exp, the Riemannian gradient.'''
    
    dim=R.shape[0]
    
    R_transpose_Eucl_grad = np.einsum('ji, jk -> ik', R, Eucl_grad) # = R{-1}*Grad f(R)
    R_transpose_Eucl_grad_vec = np.transpose(np.reshape(R_transpose_Eucl_grad,dim*dim,order='C'))

    in_exp = np.einsum('ij, j -> i' , U_U_trans, R_transpose_Eucl_grad_vec) #projecting on T_I G
    in_exp = np.reshape(in_exp, (dim, dim), order='C') #reshaping into matrix
    #print("argument in exponential is: ", in_exp)
    return in_exp #=grad f(R)

def update_R(R,x,y,U_U_trans,eta, info=False):
    '''One update step. R is the result from the previous iteration (or the initial guess).
    x are the point that should be rotated to y. U is a basis of the tangent space at identity T_I M.
    eta is the learning rate.'''
    #calculating Euclidean gradient
    Eucl_grad=func2.d_loss(R,x,y) 
    
    # retraction using QR decomposition
    #R_update, upper_tri = np.linalg.qr(R-eta*Riem_grad)
    
    #retraction using Riemanian retraction
    in_exp=func2.calc_Riem_grad(Eucl_grad,R,U_U_trans)
    R_exp = expm(-eta*in_exp)
    R_update = np.einsum('ij, jk-> ik', R, R_exp)
    
    
    if info:
        print("The new R is: ", R_update)
        print("The determinant is: ", np.linalg.det(R_update))
        print("Should be idenity: ", np.einsum("ij,kj -> ik", R_update, R_update))
        
    return R_update

def Riem_grad_descent(R_0,x,y,max_it,eta, info=False, add_init=True):
    '''Perform the Riemannian gradient descent algorithm.
    Requires: R_0, the initial guess; x, the initial image; y, the desired image;
    max_it, the maximum number of iterations; eta, the learning rate.
    Optional: info, boolean if you want extra information during iteration steps;
    add_init, boolean if True adds x to y_it at position zero.
    Output: R, final rotation matrix; y_it, all the images.'''
    #initialize end results
    n_points=x.shape[0]
    n = x.shape[1]
    y_it=np.zeros((max_it+add_init,n_points,n,1))
    
    if add_init:
        y_it[0,:,:,:]=mat_mult_points(R_0, x) #adds initial guess to y_it
    
    
    R=R_0
    
    #precompute U_U_trans
    U_U_trans=func2.construct_U_U_trans(n)

    #iteration step
    for it in range(0,max_it):

        #calculating Euclidean gradient
        R=func2.update_R(R,x,y,U_U_trans,eta, info)
        
        #print('Determinant of R is ', np.linalg.det(R))
        #print('R after iteration ',it+1)
        #print(R)
        y_it[it+add_init,:,:,:]=func2.mat_mult_points(R,x) #the initial guess is not in this
    
    return R, y_it

def construct_U_U_trans(n):
    '''Calculate U_U_trans in n dimensions'''
    U=func2.constr_skew_basis(dim=n)
    num_basis=U.shape[0]
    
    #pre-calculation
    U_matr_transpose = np.reshape(U, (num_basis, n*n), order='C')
    U_matr = np.transpose(U_matr_transpose)
    U_U_trans = np.einsum('ij, jk -> ik', U_matr, U_matr_transpose)
    
    return U_U_trans