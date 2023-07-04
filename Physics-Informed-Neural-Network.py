# Libraries and objects needed for PINN
import tensorflow as tf
import numpy as np
import math as math
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.stats.qmc import LatinHypercube
from time import time
from mpl_toolkits.mplot3d import Axes3D



# Start timer
total_time_PINN = time()



# Set data type
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)



# Set seed
tf.keras.utils.set_random_seed(0)



# Define initial conditions with respect to time
def V_Initial(S, Option = Option):
    if Option == 'Call':
        return tf.math.maximum(S - K, 0)
    
    elif Option == 'Put':    
        return tf.math.maximum(K - S, 0)



# Define boundary conditions with respect to stock price
def V_Boundary(S, tau, Option = Option):
    n = S.shape[0]
    
    if Option == 'Call':
        boundary = [0 if S[index] == 0 else s_max - K for index in range(n)]
        boundary = tf.reshape(tf.convert_to_tensor(boundary, dtype = DTYPE), [n, 1])
        return boundary
    
    elif Option == 'Put':
        boundary = [K if S[index] == 0 else 0 for index in range(n)]
        boundary = tf.reshape(tf.convert_to_tensor(boundary, dtype = DTYPE), [n, 1])
        return boundary



# Define residual of the HJB Black-Scholes equation
def Residuals(tau, S, V, V_tau, V_s, V_ss):
    return tf.math.minimum(V_tau -(((sigma ** 2) * (S ** 2) / 2) * V_ss) - ((r - d) * S * V_s) + (r * V), 
                           V - V_Initial(S))



# Set number of data points
N_0 = 1000
N_b = 500
N_r = 20000 # if using orthogonal Latin hypercube sampling must take on form p ** 2, where p is prime



# Set boundaries for time and stock price
tau_min = 0
tau_max = T
s_min = 0
s_max = K * 3



# Set lower and upper bounds for time and stock price
lb = tf.constant([tau_min, s_min], dtype = DTYPE)
ub = tf.constant([tau_max, s_max], dtype = DTYPE)



# Draw sample points for initial condition data
tau_0 = tf.ones((N_0, 1), dtype = DTYPE) * lb[0]
s_0 = tf.random.uniform((N_0,  1), lb[1], ub[1], dtype = DTYPE)
S_t_0 = tf.concat([tau_0, s_0], axis = 1)



# Evaluate initial condition with respect to time
V_0 = V_Initial(s_0)



# Draw sample points for boundary condition data
tau_b = tf.random.uniform((N_b, 1), lb[0], ub[0], dtype = DTYPE)
s_temp = tf.keras.backend.random_bernoulli((N_b, 1), 0.5, dtype = DTYPE)
s_b = np.zeros(N_b)

if Option == 'Call':
    for index in range(N_b):
        if s_temp[index] == 1:
            s_b[index] = 0
        elif s_temp[index] == 0:
            s_b[index] = s_max

elif Option == 'Put':
    for index in range(N_b):
        if s_temp[index] == 0:
            s_b[index] = 0
        elif s_temp[index] == 1:
            s_b[index] = s_max
            
s_b = tf.reshape(tf.convert_to_tensor(s_b, dtype = DTYPE), [N_b, 1])
S_t_b = tf.concat([tau_b, s_b], axis = 1)



# Evaluate boundary conditions with respect to stock price
V_b = V_Boundary(s_b, tau_b)



# Determine domain of collocation points assuming stretched grid sampling (to help PINN approximate 'kink' in solution at tau = 0)
lb_r = tf.constant([tau_min ** (0.5), s_min ** (1 / math.sqrt(2))], dtype = DTYPE)
ub_r = tf.constant([tau_max ** (0.5), s_max ** (1 / math.sqrt(2))], dtype = DTYPE)



# Draw samples using Latin hypercube sampling to be used as collocation points
sample = qmc.scale(LatinHypercube(d = 2, strength = 1).random(n = N_r), lb_r, ub_r).T
tau_r = tf.math.pow(tf.reshape(tf.convert_to_tensor(sample[0], dtype = DTYPE), [N_r, 1]), 2)
s_r = tf.math.pow(tf.reshape(tf.convert_to_tensor(sample[1], dtype = DTYPE), [N_r, 1]), math.sqrt(2))
S_t_r = tf.concat([tau_r, s_r], axis = 1)



# Collect boundary and initial data
S_t_data = [S_t_0, S_t_b]
V_data = [V_0, V_b]



#%% Initializing the model (adding 3 hidden layers, each with 20 neurons, scaling the input to be between defined stock and time bounds, 
#%%                         setting 'swish' activation function, 'glorot normal' weight initialization, and an output layer for solution output)
def Init_Model(num_hidden_layers = 3, num_neurons_per_layer = 20):
    # Initialize a feedforward neural network
    model = tf.keras.Sequential()
    
    # Input is two-dimensional (time [0, T] + stock price [0, K * 3])
    model.add(tf.keras.Input(2))
    
    # Introduce a scaling layer to map input to [[0, 0], [T, K * 3]]
    scaling_layer = tf.keras.layers.Lambda(lambda x: 2.0 * (x - lb) / (ub - lb) - 1.0)
    model.add(scaling_layer)
    
    # Append hidden layers
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer, 
                                       activation = tf.keras.activations.get('swish'),
                                       kernel_initializer = 'glorot_normal'))
        
    # Output is one-dimensional
    model.add(tf.keras.layers.Dense(1))
    
    return model



#%% A method for determining derivatives (in order to find residuals during training process)
def Get_Residuals(model, S_t_r):
    # A tf.GradientTape is used to compute derivatives in TensorFlow
    with tf.GradientTape(persistent = True) as tape:
        # Split t and s to compute partial derivatives
        tau, s = S_t_r[:, 0:1], S_t_r[:, 1:2]
        
        # Variables t and s are watched during tape to compute derivatives
        tape.watch(tau)
        tape.watch(s)
        
        # Determine residual
        V = model(tf.stack([tau[:, 0], s[:, 0]], axis = 1))
        
        # Compute gradient V_s within the GradientTape since we need second derivatives
        V_s = tape.gradient(V, s)
        
    V_tau = tape.gradient(V, tau)
    V_ss = tape.gradient(V_s, s)
    
    del tape
    
    return Residuals(tau, s, V, V_tau, V_s, V_ss)



#%% A function to determine the error in the form of residuals and improperly fit boundary/initial conditions
def Compute_Loss(model, S_t_r, S_t_data, V_data):
    # Compute residuals of each collocation point (as r)
    r = Get_Residuals(model, S_t_r)
    
    # Using mean squared error metric to determine loss, then computing the mean of the loss (as 'phi^r')
    phi_r = tf.reduce_mean(tf.square(r)) # constraint should be contained within this line
    
    # Initialize loss
    loss = phi_r
    
    # Determining loss at boundary and initial points (added together with 'phi^r' as 'loss')
    for index in range(len(S_t_data)):
        V_pred = model(S_t_data[index])
        loss += tf.reduce_mean(tf.square(V_data[index] - V_pred))
        
    return loss



#%% A method for determining the gradient of the loss function (the gradient is used to find the best way to modify the modifiable 
#%%                                                             parameters of each neuron with the goal of minimizing the loss function)
def Get_Gradient(model, S_t_r, S_t_data, V_data):
    with tf.GradientTape(persistent = True) as tape:
        # This tape is for derivatives with respect to trainable variables
        tape.watch(model.trainable_variables)
        loss = Compute_Loss(model, S_t_r, S_t_data, V_data)
    
    # Computing the gradient of the loss function
    g = tape.gradient(loss, model.trainable_variables)
    del tape
    
    return loss, g



# Initialize model, set up learning rate, and choose optimizer
model = Init_Model()
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([150, 400], [3e-1, 2e-1, 9e-2])
optim = tf.keras.optimizers.Adam(learning_rate = lr)



# Define one training step as a TensorFlow function to increase speed of training
@tf.function
def train_step():
    # Compute current loss and gradient w.r.t. trainable parameters
    loss, grad_theta = Get_Gradient(model, S_t_r, S_t_data, V_data)
    
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, model.trainable_variables))
    
    return loss



# Number of training epochs
Epochs = 450
hist = []
t_train = time()

for i in range(Epochs + 1):
    
    loss = train_step()
    
    # Append current loss to hist
    hist.append(loss.numpy())
    
    # Output current loss after 50 iterates
    if i%50 == 0:
        print('It {:05d}: loss = {:10.8e}'.format(i, loss))

        
# Print computation time
print('\nComputation time: {} seconds'.format(time() - t_train))



# Set up meshgrid
N = Accuracy
tspace = np.linspace(lb[0], ub[0], N + 1)
xspace = np.linspace(lb[1], ub[1], N + 1)
T_mesh, S_mesh = np.meshgrid(tspace, xspace)
Xgrid = np.vstack([T_mesh.flatten(), S_mesh.flatten()]).T


# Determine predictions of V_theta(tau, S)
upred = model(tf.cast(Xgrid, DTYPE))

# Reshape upred
U = upred.numpy().reshape(N + 1, N + 1)



# Surface plot of solution V_theta(tau, S)
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_ylabel("Stock Price ($)")
ax.set_xlabel("Time to Expiration (Years)")
ax.set_zlabel("American {} Premium ($)".format(Option))
ax.plot_surface(T_mesh, S_mesh, U[::-1], cmap='viridis')
ax.view_init(35, 133)
ax.set_box_aspect(aspect = None, zoom = 0.8)



print("Total computation time: {} seconds".format(time() - total_time_PINN))


