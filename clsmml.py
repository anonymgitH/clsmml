# imports
import numpy as np
from scipy.stats import t
from scipy.optimize import minimize

# Normalize and Standardize
def normalize_and_standardize(x, y):
    def normalize(x):
        Z = (x-np.min(x))/(np.max(x)-np.min(x))
        return Z
    def standardize(x):
        Z = (x-np.mean(x))/np.std(x)
        return Z 
    normalized_x = normalize(x)
    normalized_y = standardize(y)
    return normalized_x, normalized_y

# Fit data
def fit_student_t_distribution(y):
    params = t.fit(y,floc=0)
    nu_hat = params[0] 
    mu_hat = params[1] 
    tau_hat = params[2]
    return nu_hat, mu_hat, tau_hat

def compute_mles(initial_values, x, y):
    def log_likelihood(params, x, y):
        beta_0, beta, nu, tau = params
        y_pred = beta_0 + beta * x
        log_lik = np.sum(t.logpdf(y,nu, loc=y_pred, scale=tau))
        return -log_lik  
    
    bounds = [(-1, 1), (-1, 1), (200, 1000), (1e-6, np.inf)]
    result = minimize(log_likelihood, initial_values, args=(x, y), method='L-BFGS-B',bounds=bounds)
    beta0_mle, beta_mle,  nu_mle,tau_mle = result.x
    return beta0_mle, beta_mle, tau_mle, nu_mle

def calculate_khat(x, beta, tau):
    k = np.dot(beta*x.T, beta*x)
    return k

# Compute the weights wˆi
def compute_weights(n, nu, y, tau, beta_0, x, beta):
    w_hat = []    
    for i in range(n):
        delta_sq_i = ((y[i] - (beta_0 + x[i] * beta)) ** 2 )/tau
        wi = (nu + 1) / (nu + delta_sq_i)
        w_hat.append(wi)
    return w_hat

#  Compute βˆ0, βˆ
def compute_betas(params, *args):
    beta_0,beta=params
    w_i,y,x,k=args
    if  (-np.sqrt(k) - 1 <= beta <= np.sqrt(k) + 1) :
        return np.inf
    residual = (y - (beta_0 + x * beta)) ** 2
    weighted_residual = w_i * residual
    squared_weighted_residual = np.sum(weighted_residual)
    return -squared_weighted_residual

def likelihood_t(params, *args):
    tau=params
    if tau <= 0.01:
        return np.inf  
    k,nu,w_i,y,beta_0,x,beta=args
    likelihood = 0.5 * np.log(1 + ((k * (nu + 1)) / (3 * tau * (nu + 3))) ) + \
                 ((len(y) - 1) / 2) * np.log((tau)) + \
                 np.sum((w_i * (y - (beta_0 + x * beta)) ** 2) / (2 * tau))

    return -likelihood


# ALGORITHM 1
def EM_algorithm(x,y, max_iterations=100, tolerance=1e-6):
    
    normalized_x, standardize_y = normalize_and_standardize(x,y)
    
    nu_hat, mu_hat, tau_hat = fit_student_t_distribution(standardize_y)
   
    beta0_mle, beta_mle, tau_hat,_=compute_mles([1, 1, nu_hat,tau_hat],normalized_x,standardize_y)
    
    khat=calculate_khat(normalized_x, beta_mle, tau_hat)
    
    for _ in range(max_iterations):
        # E-step
        w_i_hat = compute_weights(len(standardize_y),nu_hat,standardize_y,tau_hat,beta0_mle,normalized_x,beta_mle)
        # M-step
        betas = minimize(compute_betas, [beta0_mle,beta_mle], args=(w_i_hat, standardize_y, normalized_x, khat),method='trust-constr')
        result = minimize(likelihood_t, [tau_hat], args=(khat, nu_hat,w_i_hat,standardize_y,betas.x[0],normalized_x,betas.x[1]),method='trust-constr')
        tau=result.x[0]

        if np.abs(tau_hat - tau)  < tolerance and np.abs(beta_mle-betas.x[1])  < tolerance and np.abs( beta0_mle-betas.x[0])  < tolerance :
          break
        beta_mle=betas.x[1]
        beta0_mle=betas.x[0]
        tau_hat=tau
           
    final_message_length = result.fun

    return final_message_length,beta_mle, beta0_mle, tau_hat

# ALGORITHM 2
def compute_L(x, y):
    
    def normalize(x):
        x = np.array(x)
        min_x = np.min(x)
        max_x = np.max(x)

        if min_x == max_x:
            return np.full(x.shape, 0)  
        else:
            return (x - min_x) / (max_x - min_x)
    # Normalize the data
    X = normalize(x)
    Y = normalize(y)
      
    # Step 2: Compute ηX and ηY
    pairwise_diff_X = np.abs(np.subtract.outer(X, X))
    non_zero_diff_X = pairwise_diff_X[pairwise_diff_X > 0]
    min_diff_X = np.min(non_zero_diff_X)

    pairwise_diff_Y = np.abs(np.subtract.outer(Y, Y))
    non_zero_diff_Y = pairwise_diff_Y[pairwise_diff_Y > 0]
    min_diff_Y = np.min(non_zero_diff_Y)

    # Step 3: Compute L(X) and L(Y)
    L_X = -(len(X) * np.log(min_diff_X))
    L_Y = -(len(Y) * np.log(min_diff_Y))

    return L_X, L_Y

# ALGORITHM 3
def compute_delta_X_to_Y(x, y):
  L_X, L_Y=compute_L(x,y)
  L_params,beta_mle, beta0_mle, tau_hat=EM_algorithm(x,y)
  delta_X_to_Y=(L_X+L_params)/(L_X+L_Y)
  return delta_X_to_Y

# ALGORITHM 4
def causal_direction(x, y):
    delta_X_to_Y= compute_delta_X_to_Y(x, y)
    delta_Y_to_X = compute_delta_X_to_Y(y, x)
    
    if delta_X_to_Y < delta_Y_to_X:
        direction = "->"
    elif delta_Y_to_X < delta_X_to_Y:
        direction = "<-"
    else:
        direction = "undecided"
    return direction, (-delta_X_to_Y+delta_Y_to_X)

