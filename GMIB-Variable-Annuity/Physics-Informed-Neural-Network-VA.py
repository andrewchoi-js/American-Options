# Libraries and objects needed for PINN
import tensorflow as tf
import numpy as np
import math as math
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.stats.qmc import LatinHypercube
from time import time
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
from scipy.optimize import minimize_scalar



# Set data type
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)



# Set seed
tf.keras.utils.set_random_seed(0)



# Set constants
r_act = 0.035
r_gua = 0.05
alpha_A = 0.02
alpha_G = 0.01
sigma = 0.2
w = 85 # age of policyholder when annuitized payments stop
x_0 = 65 # age of policyholder at contract inception
T = 10 # length of deferral period (T - t_0)



def phi_TC(A_tilda_values):
    n = A_tilda_values.shape[0]

    a_values = tf.reshape(tf.repeat(1.0, repeats = N_t), shape = [N_t, 1])
    
    return tf.math.maximum(A_tilda_values, a_values)



def e_integrand_gua(t_value):
    return np.exp(-r_gua * (t_value - T))

def e_integrand_act(t_value):
    return np.exp(-r_act * (t_value - T))



def phi_BC_A_tilda(t_values, A_values):
    n = A_values.shape[0]
    boundary = np.zeros(n)
    
    for index in range(n):
        
        if A_values[index] == A_min:
            boundary[index] = integrate.quad(e_integrand_gua, t_values[index], w - x_0)[0] / integrate.quad(e_integrand_act, t_values[index], w - x_0)[0]
            
        elif A_values[index] == A_max:
            boundary[index] = A_max

    return tf.convert_to_tensor(boundary, dtype = DTYPE)



# assume r in residuals in r_act
def Residuals(t, A, phi, phi_t, phi_A, phi_AA):
    return phi_t + (0.5 * (sigma ** 2) * phi_AA) + ((r_act - alpha_A) * A * phi_A) - (alpha_G * phi_A) - (r_act * phi)



# Set number of data points
N_t = 300
N_b = 300
N_r = 20000



# Set boundaries for time and A_tilda
t_min = 0
t_max = T
A_min = 0
A_max = 2



# Set lower and upper bounds for time and A_tilda
lb = tf.constant([t_min, A_min], dtype = DTYPE)
ub = tf.constant([t_max, A_max], dtype = DTYPE)



# Draw sample points for terminal condition data
t_t = tf.reshape(tf.repeat(ub[0], repeats = N_t), shape = [N_t, 1])
A_t = tf.random.uniform((N_t, 1), minval = lb[1], maxval = ub[1], dtype = DTYPE)
A_t_t_t = tf.concat([t_t, A_t], axis = 1)



# Evaluate terminal condition with respect to A_tilda
phi_t = phi_TC(A_t)



# Draw sample points for boundary condition data
t_b = tf.random.uniform((N_b, 1), minval = lb[0], maxval = ub[0], dtype = DTYPE)
A_b = tf.keras.backend.random_bernoulli((N_b, 1), 0.5, dtype = DTYPE) * A_max
# A_b_temp = tf.keras.backend.random_bernoulli((N_b, 1), 0.5, dtype = DTYPE) * A_max
# A_b = tf.zeros([N_b, 1], dtype = DTYPE)
A_b_t_b = tf.concat([t_b, A_b], axis = 1)



phi_b = phi_BC_A_tilda(t_b, A_b)



# Determine domain of collocation points
lb_r = tf.constant([t_min, A_min], dtype = DTYPE)
ub_r = tf.constant([t_max, A_max], dtype = DTYPE)


# Draw samples using Latin hypercube sampling to be used as collocation points
sample = qmc.scale(LatinHypercube(d = 2).random(n = N_r), lb_r, ub_r).T
t_r = tf.reshape(tf.convert_to_tensor(sample[0], dtype = DTYPE), [N_r, 1])
A_r = tf.reshape(tf.convert_to_tensor(sample[1], dtype = DTYPE), [N_r, 1])
A_r_t_r = tf.concat([t_r, A_r], axis = 1)



# Collect boundary and terminal data
A_t_data = [A_t_t_t, A_b_t_b]
phi_data = [phi_t, phi_b]



#%%
def Init_Model(num_hidden_layers = 3, num_neurons_per_layer = 20):
    # Initialize a feedforward neural network
    model = tf.keras.Sequential()
    
    # Input is two-dimensional (time + one spatial dimension)
    model.add(tf.keras.Input(2))
    
    # Introduce a scaling layer to map input to [lb, ub]
    scaling_layer = tf.keras.layers.Lambda(lambda x: 2.0 * (x - lb) / (ub - lb) - 1.0)
    model.add(scaling_layer)
    
    # Append hidden layers
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer, 
                                       activation = tf.keras.activations.get('swish'),
                                       #activation = tf.keras.layers.LeakyReLU(alpha = 0.1), 
                                       #activation = tf.keras.activations.get('swish'),
                                       kernel_initializer = 'glorot_normal'))
        
    # Output is one-dimensional
    model.add(tf.keras.layers.Dense(1))
    
    return model



#%%
def Get_Residuals(model, A_r_t_r):
    # A tf.GradientTape is used to compute derivatives in TensorFlow
    with tf.GradientTape(persistent = True) as tape:
        # Split t and s to compute partial derivatives
        t, A = A_r_t_r[:, 0:1], A_r_t_r[:, 1:2]
        
        # Variables t and s are watched during tape to compute derivatives
        tape.watch(t)
        tape.watch(A)
        
        # Determine residual
        phi = model(tf.stack([t[:, 0], A[:, 0]], axis = 1))
        
        # Compute gradient V_s within the GradientTape since we need second derivatives
        phi_A = tape.gradient(phi, A)
        
    phi_t = tape.gradient(phi, t)
    phi_AA = tape.gradient(phi_A, A)
    
    del tape
    
    return Residuals(t, A, phi, phi_t, phi_A, phi_AA), phi



#%%
def Compute_Loss(model, A_r_t_r, A_t_data, phi_data):
    # Compute phi^r
    r, phi = Get_Residuals(model, A_r_t_r)
    #payoff = tf.nn.relu(K - A_r_t_r[1])
    phi_r = tf.reduce_mean(tf.square(r)) # constraint should be contained within this line
    #phi_r += tf.reduce_mean(tf.square(tf.math.maximum(V - payoff, 0)))
    
    # Initialize loss
    loss = phi_r
    
    # Add phi^0 and phi^b to the loss
    for index in range(len(A_t_data)):
        phi_pred = model(A_t_data[index])
        loss += tf.reduce_mean(tf.square(phi_data[index] - phi_pred))
        
    return loss



#%%
def Get_Gradient(model, A_r_t_r, A_t_data, phi_data):
    with tf.GradientTape(persistent = True) as tape:
        # This tape is for derivatives with respect to trainable variables
        tape.watch(model.trainable_variables)
        loss = Compute_Loss(model, A_r_t_r, A_t_data, phi_data)
        
    g = tape.gradient(loss, model.trainable_variables)
    del tape
    
    return loss, g



# Initialize model, set up learning rate, and choose optimizer
model = Init_Model()
#lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([150, 600], [1e-1, 7e-2, 2e-4])
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([200, 1150], [3e-1, 1e-1, 6e-2])
#lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([250, 400, 3000], [1e-1, 7e-2, 1e-2, 2e-4])
optim = tf.keras.optimizers.Adam(learning_rate = lr, use_ema = True, ema_momentum = 0.9)#, ema_overwrite_frequency = 1) 



# Define one training step as a TensorFlow function to increase speed of training
@tf.function
def train_step():
    # Compute current loss and gradient w.r.t. parameters
    loss, grad_theta = Get_Gradient(model, A_r_t_r, A_t_data, phi_data)
    
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, model.trainable_variables))
    
    return loss

# Number of training epochs
N_train = 250
hist = []


# Start timer
t0 = time()

for i in range(N_train + 1):
    
    loss = train_step()
    
    # Append current loss to hist
    hist.append(loss.numpy())
    
    # Output current loss after 50 iterates
    if i%50 == 0:
        print('It {:05d}: loss = {:10.8e}'.format(i,loss))

        
# Print computation time
print('\nComputation time: {} seconds'.format(time()-t0))



# Set up meshgrid
N = 200
tspace = np.linspace(lb[0], ub[0], N + 1)
xspace = np.linspace(lb[1], ub[1], N + 1)
T, X = np.meshgrid(tspace, xspace)
Xgrid = np.vstack([T.flatten(),X.flatten()]).T

# Determine predictions of u(t, x)
upred = model(tf.cast(Xgrid,DTYPE))

# Reshape upred
U = upred.numpy().reshape(N+1,N+1)


# determine index of 1 in [A_min, A_max]
difference_array = np.abs(xspace - 1)
index = difference_array.argmin()



def negative_phi(gamma, phi, row, column):
    ## find a way to avoid dividing by 0
    if column == 0:
        ## changed maximum to minimum bc im not sure but it seems to work
        return - (np.minimum(0.000001 - gamma, 1 - (gamma / 0.000001)) * phi[row][index if 1 < 0.000001 else column] + gamma)
    else:
        return - (np.minimum(xspace[column] - gamma, 1 - (gamma / xspace[column])) * phi[row][index if 1 < xspace[column] else column] + gamma)



def phi_JC():
    phi = U.T
    gamma = [[minimize_scalar(fun = negative_phi, bounds = (A_min, A_max), args = (phi, row, column)).x for row in range(N + 1)] for column in range(N + 1)]
    phi_jc = [[minimize_scalar(fun = negative_phi, bounds = (A_min, A_max), args = (phi, row, column)) for row in range(N + 1)] for column in range(N + 1)]
    return gamma, phi_jc 



Z = np.array(phi_JC()[0])



def Heat_Map_PINN():
  t_plot, A_plot = np.meshgrid(tspace, xspace)
  plt.pcolormesh(t_plot, A_plot, Z)
  plt.colorbar()
  plt.show()

Heat_Map_PINN()
