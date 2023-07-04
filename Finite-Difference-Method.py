# Set constants
Option = "Put"
K = 24
T = 0.4
sigma = 0.38
r = 0.02
d = 0.005
Accuracy = 300
True_Accuracy = 1000
Stock_Price = 23
exact = False


# Libraries and objects needed for finite difference method
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg
from scipy import interpolate
from scipy.linalg import solve_banded
from time import time
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# Beginning timer to approximate total run time
t_total_FDM = time()

### Why are we using difference s_max and S_max??



#%% Outputs a 2D matrix containing American Option premium values w.r.t. stock price and time to expiration

# Beginning timer
t_FDM = time()

def American_Option_FD(Option = Option, K = K, T = T, sigma = sigma, r = r, d = d, exact = exact):
    
    """
        
    Parameters
    ----------
    
    Option : "Call" or "Put"
    K : Strike Price ($)
    T : Time to expiration (Years)
    sigma : Volatility of underlying asset (Input in decimal form)
    r : Risk free interest rate (Input in decimal form)
    d : Continuous dividend yield of underlying asset (Input in decimal form)
    Accuracy: Specify the desired number grid points in time and stock price (should range between 200 and 1500)
    
    """
        
    def Put_Payoff(m):
        return max(K - (m * dS), 0)
    
    def Call_Payoff(m):
        return max((m * dS) - K, 0)
    
    if exact == False:
        M = Accuracy
        N = Accuracy
    else:
        M = True_Accuracy
        N = True_Accuracy
    
    S_max = K * 3
    dS = S_max / M
    S = np.linspace(0, S_max, num = M + 1)
    
    dt = T / N
    t = np.linspace(0, T, num = N + 1)
      
    P = np.zeros((M + 1, N + 1))
    X = np.zeros((M + 1, N + 1))
    f = np.zeros((M + 1, N + 1))
    
    # Filling matrices P, X with matrix coefficients
    for m in range(1, M):
        P[m][m] = 1 + (0.5 * r * dt) + (0.5 * (sigma ** 2) * (m ** 2) * dt)
        P[m][m + 1] = (-0.25 * (sigma ** 2) * (m ** 2) * dt) - (0.25 * (r - d) * m * dt)
        P[m + 1][m] = (0.25 * (r - d) * (m + 1) * dt) - (0.25 * (sigma ** 2) * ((m + 1) ** 2) * dt)
        
        X[m][m] = 1 - (0.5 * r * dt) - (0.5 * (sigma ** 2) * (m ** 2) * dt)
        X[m][m + 1] = (0.25 * (r - d) * m * dt) + (0.25 * (sigma ** 2) * (m ** 2) * dt)
        X[m + 1][m] = (0.25 * (sigma ** 2) * ((m + 1) ** 2) * dt) - (0.25 * (r - d) * (m + 1) * dt)
    
    # Satisfying boundary/initial constraints listed at the bottom of page 73
    P[0][0] = 1
    P[-1][-1] = 1
    P[1][0] = (((r - d) * dt)  / (4 * ((r * dt) - 1))) - (((sigma ** 2) * dt) / (4 * ((r * dt) - 1)))
    P[-1][-2] = 0
    
    X[0][0] = 1
    X[-1][-1] = 1
    X[1][0] = (((sigma ** 2) * dt) / (4 * ((r * dt) - 1))) - (((r - d) * dt) / (4 * (1 + (r * dt))))
    X[-1][-2] = 0   
    
    # Filling solution matrix f with initial and boundary conditions
    if Option == "Put":
        for n in range(N + 1):
            for m in range(M + 1):
                if m == 0:
                    f[n][0] = K
                elif m == M:
                    f[n][-1] = 0
                elif n == N:
                    f[-1][m] = Put_Payoff(m)
                    
    elif Option == "Call":
        for n in range(N + 1):
            for m in range(M + 1):
                if m == 0:
                    f[n][0] = 0
                elif m == M:
                    f[n][-1] = S_max - K
                elif n == N:
                    f[-1][m] = Call_Payoff(m)


# In the process of implementing banded matrix solver for faster run time
#     upper_P = np.insert((np.diag(P, k = 1)), 0, 0)
#     middle_P = np.diag(P, k = 0)
#     lower_P = np.append((np.diag(P, k = -1)), 0)
#     stacked_P = np.vstack((upper_P, middle_P, lower_P))
    
#     upper_X = np.insert((np.diag(X, k = 1)), 0, 0)
#     middle_X = np.diag(X, k = 0)
#     lower_X = np.append((np.diag(X, k = -1)), 0)
#     stacked_X = np.vstack((upper_X, middle_X, lower_X))
    
#     for n in range(N - 1, -1, -1):
#         f[n][1:M] = solve_banded((1, 1), stacked_P, solve_banded((1, 1), stacked_X, f[n + 1]))[1:M]


    # Solving for time steps n = N -1, N - 2, ..., 0 and incorporating maximum condition
    for n in range(N - 1, -1, -1):
        f[n][1:M] = np.linalg.solve(P, np.matmul(X, f[n + 1]))[1:M]
        
        if Option == "Put":
            for m in range(1, M + 1):
                f[n][m] = max(f[n][m], Put_Payoff(m))
        elif Option == "Call":
            for m in range(1, M + 1):
                f[n][m] = max(f[n][m], Call_Payoff(m))
    
    # Solution was obtained backwards in time, must reverse solution rows to obtain solution forward in time
    f = f[::-1]
    
    return f



American_Option_FD()
print("Computation time of FDM matrix solver: {} seconds".format(time() - t_FDM))
print()
print()



#%% Outputs the premium of an American option w.r.t. the below parameters 

# Beginning timer
t_premium = time()

def American_Option_FD_Premium(Option = Option, K = K, T = T, sigma = sigma, r = r, d = d, Stock_Price = Stock_Price, exact = exact):
    
    """
        
    Parameters
    ----------
    
    Option : "Call" or "Put"
    K : Strike Price ($)
    T : Time to expiration (Years)
    sigma : Volatility of underlying asset (Input in decimal form)
    r : Risk free interest rate (Input in decimal form)
    d : Continuous dividend yield of underlying asset (Input in decimal form) 
    Stock_Price: Price of stock at t = 0
    Accuracy: Specify the desired number of time and space steps (should range between 200 and 1500)
    
    """

    M = Accuracy

    S_max = K * 3
    dS = S_max / M
    S = np.linspace(0, S_max, num = M + 1)

    interp = interpolate.splrep(S, American_Option_FD(Option = Option, K = K, T = T, sigma = sigma, r = r, d = d, exact = exact)[-1])
    
    print("The {} premium is ${:.4f} assuming a strike price of ${}, {} year(s) until expiration, a volatility of {:.0f}%, a risk free " \
    "interest rate of {:.0f}%, an underlying asset value of ${}, and a continuous dividend yield of {:.0f}%.".format(Option, 
                                                                                                                     interpolate.splev(Stock_Price, interp),
                                                                                                                     K,
                                                                                                                     T,
                                                                                                                     sigma,
                                                                                                                     r,
                                                                                                                     Stock_Price,
                                                                                                                     d))

          
          
American_Option_FD_Premium()
print("\nComputation time of premium interpolation: {} seconds".format(time() - t_premium))
print()



# #%% Outputs the free boundary of an American option w.r.t. the below parameters

# Beginning timer
t_FB = time()

def Free_Boundary(Option = Option, K = K, T = T, sigma = sigma, r = r, d = d, exact = exact):
          
    """
        
    Parameters
    ----------
    
    Option : "Call" or "Put"
    K : Strike Price ($)
    T : Time to expiration (Years)
    sigma : Volatility of underlying asset (Input in decimal form)
    r : Risk free interest rate (Input in decimal form)
    d : Continuous dividend yield of underlying asset (Input in decimal form) 
    Accuracy: Specify the desired number of time and space steps (should range between 200 and 1500)
    
    """

    M = Accuracy
    N = Accuracy

    if Option == "Call":
        American_Call = American_Option_FD(Option = "Call", K = K, T = T, sigma = sigma, r = r, d = d, exact = exact)
    elif Option == "Put":
        American_Put = American_Option_FD(Option = "Put", K = K, T = T, sigma = sigma, r = r, d = d, exact = exact)
    
    S_max = K * 3
    S_f = np.zeros((N + 1))
    dS = S_max / M
    S = np.linspace(0, S_max, num = M + 1)
    
    dt = T / N
    t = np.linspace(0, T, num = N + 1)
    
    for n in range(N + 1):
        for m in range(M + 1):
            
            if Option == "Call":
                if American_Call[n][m] <= S[m] - K:
                    S_f[n] = S[m]
                else:
                    pass
                
            elif Option == "Put":
                if American_Put[n][m] <= K - S[m]:
                    S_f[n] = S[m]
                else:
                    pass
                
    plt.plot(t, S_f)
    plt.show()



Free_Boundary()
print("\nComputation time to determine free boundary: {} seconds".format(time() - t_FB))
print()



#%% Outputs a 2D plot of an American option"s premium w.r.t. the below parameters

# Beginning timer
t_2D = time()

def Plot_2D_FD(Option = Option, exact = False):

    M = Accuracy
    N = Accuracy
    S_max = K * 3
    S = np.linspace(0, S_max, num = M + 1)
    
    American_Option = American_Option_FD(Option = Option, K = K, T = T, sigma = sigma, r = r, d = d, exact = exact)
          
    for index in range(N + 1):
        plt.plot(S, American_Option[index])

    plt.show()

          
          
Plot_2D_FD()
print("\nComputation time to construct and output 2D plot: {} seconds".format(time() - t_2D))
print()



#%% Outputs a 3D plot of an American option"s premium w.r.t. the below parameters

# Beginning timer
t_3D = time()

def Plot_3D_FD(Option = Option, K = K, T = T, sigma = sigma, r = r, d = d, exact = exact):
    
    """
        
    Parameters
    ----------
    
    Option : "Call" or "Put"
    K : Strike Price ($)
    T : Time to expiration (Years)
    sigma : Volatility of underlying asset (Input in decimal form)
    r : Risk free interest rate (Input in decimal form)
    d : Continuous dividend yield of underlying asset (Input in decimal form) 
    Accuracy: Specify the desired number of time and space steps (should range between 200 and 1500)
    
    """
    
    if exact == False:
        M = Accuracy
        N = Accuracy
    else:
        M = True_Accuracy
        N = True_Accuracy

    S_max = K * 3
    dS = S_max / M
    S = np.linspace(0, S_max, num = M + 1)
    
    dt = T / N
    t = np.linspace(0, T, num = N + 1)
    
    if Option == "Call":
        American_Option = American_Option_FD(Option = "Call", K = K, T = T, sigma = sigma, r = r, d = d, exact = exact)
    elif Option == "Put":
        American_Option = American_Option_FD(Option = "Put", K = K, T = T, sigma = sigma, r = r, d = d, exact = exact)
    
    hf = plt.figure()
    ha = hf.add_subplot(111, projection = "3d")
    X, T = np.meshgrid(S, t)
    ha.plot_surface(X, T, American_Option, cmap = "viridis")
    ha.set_xlabel("Stock Price ($)")
    ha.set_ylabel("Time to Expiration (Years)")
    ha.set_zlabel("American {} Premium ($)".format(Option))
    plt.title('FDM American Option Solution (3D)')
    ha.set_box_aspect(aspect = None, zoom = 0.8)
    
    if Option == "Put":
        ha.set_zlim(0, K)
        ha.view_init(30, -38)
        plt.draw()
        plt.pause(0.00001)
        plt.show()

    elif Option == "Call":
        ha.set_zlim(0, S_max - K)
        ha.view_init(30, -138)
        plt.draw()
        plt.pause(0.00001)
        plt.show()



Plot_3D_FD()
print("\nComputation time to construct and output 3D plot: {} seconds".format(time() - t_3D))
print()
print()

print("Total computation time: {} seconds".format(time() - t_total_FDM))


