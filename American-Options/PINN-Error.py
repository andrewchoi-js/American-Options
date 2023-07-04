#%% Computing difference between PINN and FDM with fine mesh grid (approx. exact solution)
def PINN_Error(K = K, T = T, sigma = sigma, r = r, d = d, Option = Option, exact = True):
    
    S_max = K * 3
    dS = S_max / True_Accuracy
    S = np.linspace(0, S_max, num = True_Accuracy + 1)
    
    dt = T / True_Accuracy
    t = np.linspace(0, T, num = True_Accuracy + 1)
    American_Option = American_Option_FD(Option = Option, K = K, T = T, sigma = sigma, r = r, d = d, exact = exact)

    tspace = np.linspace(lb[0], ub[0], True_Accuracy + 1)
    xspace = np.linspace(lb[1], ub[1], True_Accuracy + 1)
    T, X = np.meshgrid(tspace, xspace)
    Xgrid = np.vstack([T.flatten(), X.flatten()]).T
    
    # Determine predictions of u(t, x)
    upred = model(tf.cast(Xgrid, DTYPE))
    
    # Reshape upred
    U = upred.numpy().reshape(True_Accuracy + 1, True_Accuracy + 1)
    
    Error = tf.math.abs(U - American_Option.T)
    
    hf = plt.figure()
    ha = hf.add_subplot(111, projection = '3d')
    X, T = np.meshgrid(S, t)

    
    ha.plot_surface(X, T, Error, cmap = 'viridis')
    plt.show()


    
PINN_Error()

