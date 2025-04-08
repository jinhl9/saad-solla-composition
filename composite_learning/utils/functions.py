import numpy as np
from scipy.optimize import fsolve

def gram_schmidt(N, K):
    """
    Given the dimension space dimension N, generate K random vectors and its orthogonal spans
    """

    def proj(u, v):
        """
        Return projection of v to u
        """
        return np.dot(v, u) / np.dot(u, u) * u

    V = np.random.normal(loc=0., scale=1., size=(K, N))
    U = np.zeros_like(V)

    ## Initialise u1 to v1
    U[0] = V[0]

    ## Gram-schomidt process
    for k in range(1, K):
        projection_terms = [proj(U[i], V[k]) for i in range(k)]
        U[k] = V[k] - np.sum(projection_terms, axis=0)

    return V, U

def control_VS(VT, angle, positive=True, max_iter=int(1e+5)):
    """
    Given the vector, return the vector rotated with 'angle'.
    """
    dim = len(VT)
    VT_norm = VT / np.linalg.norm(VT)
    abs_flag = False
    count =0
    if positive:
        while not abs_flag and max_iter > count:
            count +=1
            a = np.random.normal(loc=0., scale=0.1, size=(dim))
            b = np.random.normal(loc=0., scale=0.1, size=(dim))
            h = (b - a) - np.dot((b - a), VT_norm) * VT_norm
            v = np.cos(angle) * VT_norm + np.sin(angle) * h / np.linalg.norm(h)
            if all(v>=0):
                abs_flag = True
            if max_iter == count:
                print('Could not find a positive vector within in max_iter')
                return None
    else:
        a = np.random.normal(loc=0., scale=10, size=(dim))
        b = np.random.normal(loc=0., scale=10, size=(dim))
        h = (b - a) - np.dot((b - a), VT_norm) * VT_norm
        v = np.cos(angle) * VT_norm + np.sin(angle) * h / np.linalg.norm(h)

        
    return v

def primitive_training_approx(timesteps, T, N, eta = 1):
    t_term = eta/np.sqrt(2*np.pi)/np.pi*(2-T)/N
    rho_list = np.zeros(len(timesteps))
    
    for i,t in enumerate(timesteps):
        rho=-np.pi/2 + np.pi*np.power(t_term*t + np.power(2,T-2), 1/(2-T))
        rho_list[i] = rho

    critical_t = np.sqrt(2*np.pi)*np.pi/eta/(T-2)*np.power(2, (T-2))*N

    return rho_list, critical_t

def max_overlap(T):
    def _max_overlap_numerical(x, T):
        return 1/T*x*(1-1/np.pi*np.arccos(x)) - np.sqrt(2/np.pi)*(1-x**2)
    x_init = 0.9
    return fsolve(_max_overlap_numerical, x_init, args=(T)) - 0.01

def compositional_generalisation(timesteps, VS0, VT0, T,  R, N):
    rho_list = []
    rho0 = np.dot(VS0, VT0)*R
    alpha = 1-np.arccos(rho0)/np.pi
    beta = 1/np.pi/np.sqrt(1-rho0**2)
    gamma = 1/np.sqrt(2*np.pi)/N
    delta_sq = np.sum(VS0**2)*R**2
    epsilon = (T-1)*beta/alpha
    m = delta_sq + rho0**2
    n = 1-epsilon*rho0
    a = -2*rho0/m
    b = epsilon/n
    c1 = (1+rho0*b)/(1+a*rho0)
    for t in timesteps:
        y = np.exp((b-a)*gamma*np.power(alpha, T-1)*m*n*t)
        rho_list.append(((c1*y-1)/(b-a*c1*y)).item())

    t_sat = np.power(alpha, 1-T) * np.log((1+b*np.sqrt(delta_sq))/c1/(1+a*np.sqrt(delta_sq)))/(b-a)/gamma/m/n
    return rho_list, t_sat

    